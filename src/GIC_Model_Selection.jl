##############################################################################
#  GIC_Model_Selection.jl
#
#  Hopfield-style coordinate search over variable subsets, maximizing a GIC.
#  Requires penalty_selection.jl to be loaded first.
#
#  For the built-in criteria the search never refits from scratch: it keeps
#  M = (X_S'X_S)^{-1}, B = X_S'Y and Y'HY for the current subset and scores a
#  candidate move in O(nK + K^2) with rank-one updates:
#
#    add z:     u = X_S'x_z,  s = x_z'x_z - u'Mu,  e = x_z'Y - B'Mu
#               Y'HY_new = Y'HY + e e' / s
#    remove j:  beta_j = (M B)[j, :],   Y'HY_new = Y'HY - beta_j' beta_j / M_jj
#
#  M itself is only updated when a move is accepted. A user-supplied criterion
#  function that is not one of the built-ins is called directly instead.
##############################################################################
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Incremental least-squares state
# ---------------------------------------------------------------------------
mutable struct _SubsetFit
    S::Vector{Int}              # selected columns, in insertion order
    Xs::Matrix{Float64}         # X[:, S]
    M::Matrix{Float64}          # (Xs'Xs)^{-1}
    B::Matrix{Float64}          # Xs'Y   (K x q)
    YHY::Matrix{Float64}        # Y'HY   (q x q)
    logdetG::Float64            # logdet(Xs'Xs), for CAICF
    fitted::Matrix{Float64}     # H Y (n x q), only kept for AttIC
    accepted::Int               # moves since the last full refactorization
end

struct _SearchData
    X::Matrix{Float64}
    Y::Matrix{Float64}          # n x q
    Yorig                       # the response as given (vector or matrix)
    XtY::Matrix{Float64}        # p x q
    colsq::Vector{Float64}      # ||x_j||^2
    YtY::Matrix{Float64}        # q x q
    crit::Symbol
    P::Int
    gamma::Float64
end

_as_response(A::Matrix{Float64}, Yorig) = Yorig isa AbstractVector ? A[1, 1] : A
_as_fitted(F::Matrix{Float64}, Yorig) = Yorig isa AbstractVector ? vec(F) : F

function _SearchData(X, Y, crit::Symbol, P::Integer, gamma::Real)
    Xf = Matrix{Float64}(X)
    Ym = Y isa AbstractVector ? reshape(Vector{Float64}(Y), :, 1) : Matrix{Float64}(Y)
    _SearchData(Xf, Ym, Y, Xf' * Ym, vec(sum(abs2, Xf; dims = 1)), Ym' * Ym,
                crit, Int(P), Float64(gamma))
end

function _refit(d::_SearchData, S::Vector{Int})
    Xs = d.X[:, S]
    M, logdetG = _gram_inverse(Xs' * Xs)
    B = d.XtY[S, :]
    YHY = Matrix{Float64}(B' * (M * B))
    fitted = _needs_fitted(d.crit) ? Xs * (M * B) : zeros(0, 0)
    _SubsetFit(copy(S), Xs, Matrix{Float64}(M), Matrix{Float64}(B), YHY,
               logdetG, fitted, 0)
end

function _score(d::_SearchData, YHY::Matrix{Float64}, K::Int, logdetG::Float64,
                fitted)
    T = size(d.X, 1)
    _gic_value(d.crit, _as_response(YHY, d.Yorig), _as_response(d.YtY - YHY, d.Yorig),
               T, K, d.P; Y = d.Yorig,
               fitted = fitted === nothing ? nothing : _as_fitted(fitted, d.Yorig),
               logdetG = logdetG, gamma = d.gamma)
end

_score(d::_SearchData, f::_SubsetFit) =
    _score(d, f.YHY, length(f.S), f.logdetG,
           _needs_fitted(d.crit) ? f.fitted : nothing)

# Score of adding column z. Returns (value, cache) or (nothing, nothing) if
# x_z is (numerically) in the span of the current columns.
function _try_add(d::_SearchData, f::_SubsetFit, z::Int)
    xz = view(d.X, :, z)
    K = length(f.S)
    if K == 0
        s = d.colsq[z]
        Mu = Float64[]
        e = d.XtY[z, :]
    else
        u = f.Xs' * xz
        Mu = f.M * u
        s = d.colsq[z] - dot(u, Mu)
        e = d.XtY[z, :] - f.B' * Mu
    end
    s <= 1e-10 * max(d.colsq[z], eps()) && return (nothing, nothing)
    YHY = f.YHY + (e * e') ./ s
    logdetG = f.logdetG + log(s)
    fitted = nothing
    if _needs_fitted(d.crit)
        r = K == 0 ? Vector(xz) : xz - f.Xs * Mu      # residual of x_z on X_S
        fitted = f.fitted .+ r * (e' ./ s)
    end
    return (_score(d, YHY, K + 1, logdetG, fitted), (Mu, s, YHY, logdetG, fitted))
end

function _accept_add!(d::_SearchData, f::_SubsetFit, z::Int, cache)
    Mu, s, YHY, logdetG, fitted = cache
    K = length(f.S)
    Mnew = Matrix{Float64}(undef, K + 1, K + 1)
    Mnew[1:K, 1:K] = f.M .+ (Mu * Mu') ./ s
    Mnew[1:K, K + 1] = -Mu ./ s
    Mnew[K + 1, 1:K] = -Mu ./ s
    Mnew[K + 1, K + 1] = 1 / s
    f.M = Mnew
    push!(f.S, z)
    f.Xs = hcat(f.Xs, view(d.X, :, z))
    f.B = vcat(f.B, d.XtY[z:z, :])
    f.YHY = YHY
    f.logdetG = logdetG
    fitted === nothing || (f.fitted = fitted)
    f.accepted += 1
    return f
end

# Score of removing the column at position j of f.S.
function _try_remove(d::_SearchData, f::_SubsetFit, j::Int)
    mjj = f.M[j, j]
    betaj = vec(f.M[j, :]' * f.B)                     # coefficient row j
    YHY = f.YHY - (betaj * betaj') ./ mjj
    logdetG = f.logdetG + log(mjj)
    fitted = nothing
    if _needs_fitted(d.crit)
        fitted = f.fitted .- (f.Xs * f.M[:, j]) * (betaj' ./ mjj)
    end
    return (_score(d, YHY, length(f.S) - 1, logdetG, fitted), (YHY, logdetG, fitted))
end

function _accept_remove!(d::_SearchData, f::_SubsetFit, j::Int, cache)
    YHY, logdetG, fitted = cache
    keep = [i for i in eachindex(f.S) if i != j]
    mj = f.M[keep, j]
    f.M = f.M[keep, keep] .- (mj * mj') ./ f.M[j, j]
    deleteat!(f.S, j)
    f.Xs = f.Xs[:, keep]
    f.B = f.B[keep, :]
    f.YHY = YHY
    f.logdetG = logdetG
    fitted === nothing || (f.fitted = fitted)
    f.accepted += 1
    return f
end

# The subset in the order the original implementation reported it (new
# variables are prepended), so outputs are unchanged.
_report(f::_SubsetFit) = reverse(f.S)

"""
    GIC_Variable_Selection(X, Y, Init_Columns, Calculate_GIC, Calculate_GIC_short, k;
                           Nsim = 1, gamma = 0.6)

Variable selection by greedy coordinate search: visit every column in random
order `Nsim` times; add it if absent, remove it if present, and keep the move
if the criterion increases. A move that would give a model with `k` or more
columns is skipped, and models never exceed n - 1 columns (n = sample size),
so the selected model has at most min(k, n) - 1 variables.

`Calculate_GIC` is one of the built-in `Calculate_*` functions (fast path) or
any function `(Y, X_subset, P) -> (value, Inverse)`. `Calculate_GIC_short` is
accepted for backward compatibility and is not needed by the fast path.
`gamma` is the EBIC tuning constant.

Returns `(GIC_list, GIC_coeff, GIC_time)`: the criterion trace, the selected
column indices, and the elapsed time in seconds since the search started,
after each step.
"""
function GIC_Variable_Selection(
    X::AbstractMatrix,
    Y::Union{AbstractVector, AbstractMatrix},
    Init_Columns::AbstractVector{<:Integer},
    Calculate_GIC,
    Calculate_GIC_short,
    k::Integer;
    Nsim::Integer = 1,
    gamma::Real = 0.6
)
    t0 = time_ns()
    m, p = size(X)
    kcap = min(k, m)          # models have at most min(k, n) - 1 variables
    @assert length(Init_Columns) ≤ p "Initial columns exceed design matrix size."
    @assert m == size(Y, 1) "Sample size mismatch between X and Y."

    visit = vcat([shuffle(collect(1:p)) for _ in 1:Nsim]...)
    crit = _criterion_symbol(Calculate_GIC)
    crit === nothing &&
        return _GIC_Variable_Selection_generic(X, Y, Init_Columns, Calculate_GIC, kcap, visit, t0)

    d = _SearchData(X, Y, crit, p, gamma)
    f = _refit(d, reverse(collect(Int, Init_Columns)))
    current = _score(d, f)
    inS = falses(p)
    inS[f.S] .= true

    GIC_list = zeros(length(visit))
    GIC_coeff = Vector{Any}(undef, length(visit))    # Vector{Any} -> plain R list
    GIC_time = zeros(length(visit))

    for (w, z) in enumerate(visit)
        if inS[z]
            j = findfirst(==(z), f.S)
            value, cache = _try_remove(d, f, j)
            if tr(current) < tr(value)
                _accept_remove!(d, f, j, cache)
                inS[z] = false
                current = value
            end
        elseif length(f.S) + 1 < kcap
            value, cache = _try_add(d, f, z)
            if value !== nothing && tr(current) < tr(value)
                _accept_add!(d, f, z, cache)
                inS[z] = true
                current = value
            end
        end
        if f.accepted >= 200          # refactorize now and then to avoid drift
            f = _refit(d, f.S)
            current = _score(d, f)
        end
        GIC_list[w] = tr(current)
        GIC_coeff[w] = _report(f)
        GIC_time[w] = (time_ns() - t0) / 1e9
    end
    return (GIC_list, GIC_coeff, GIC_time)
end

# Fallback for user-supplied criterion functions: refit each candidate.
function _GIC_Variable_Selection_generic(X, Y, Init_Columns, Calculate_GIC, kcap, visit,
                                         t0 = time_ns())
    p = size(X, 2)
    S = collect(Int, Init_Columns)
    current, _ = Calculate_GIC(Y, X[:, S], p)
    GIC_list = zeros(length(visit))
    GIC_coeff = Vector{Any}(undef, length(visit))    # Vector{Any} -> plain R list
    GIC_time = zeros(length(visit))
    for (w, z) in enumerate(visit)
        idx = findfirst(==(z), S)
        cand = idx === nothing ? (length(S) + 1 < kcap ? vcat(z, S) : nothing) :
                                 deleteat!(copy(S), idx)
        if cand !== nothing
            value, _ = Calculate_GIC(Y, X[:, cand], p)
            if tr(current) < tr(value)
                current, S = value, cand
            end
        end
        GIC_list[w] = tr(current)
        GIC_coeff[w] = copy(S)
        GIC_time[w] = (time_ns() - t0) / 1e9
    end
    return (GIC_list, GIC_coeff, GIC_time)
end

# Entry point for the R wrapper. The criteria are passed by name: a Julia
# function handed to R and back arrives as an RCall.RFunction wrapper, which
# the search cannot recognise, so it would fall back to the slow generic path
# and call back into R for every candidate. The selected sets come back packed
# into one integer vector plus their lengths, which converts to R in one step.
_julia_function(name::AbstractString) = getfield(Main, Symbol(name))

function _GIC_Variable_Selection_flat(X, Y, Init_Columns, Calculate_GIC::AbstractString,
                                      Calculate_GIC_short::AbstractString, k; kwargs...)
    GIC_list, GIC_coeff, GIC_time = GIC_Variable_Selection(
        X, Y, Init_Columns, _julia_function(Calculate_GIC),
        _julia_function(Calculate_GIC_short), k; kwargs...)
    lens = Int[length(s) for s in GIC_coeff]
    flat = Vector{Int}(undef, sum(lens))
    i = 0
    for s in GIC_coeff, j in s
        flat[i += 1] = j
    end
    return (GIC_list, flat, lens, GIC_time)
end
