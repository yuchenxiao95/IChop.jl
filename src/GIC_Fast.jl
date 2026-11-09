# Fast path for GIC_Variable_Selection (ported from GICHighDimension, inst/julia).
#
# Every criterion below depends on the data only through
#     Y'HY = b' M b,   RSS = Y'Y - Y'HY,   b = X_S'Y,   M = (X_S'X_S)^{-1},
# so the n x n hat matrix is never formed. The search keeps M and b for the
# current model and scores each proposed move with a rank-one update:
#   add x_z:     u = X_S'x_z, s = x_z'x_z - u'Mu, e = x_z'Y - b'Mu,
#                Y'HY -> Y'HY + e e' / s
#   remove j:    beta_j = (M b)_j,  Y'HY -> Y'HY - beta_j beta_j' / M_jj
# Scoring a move costs O(nK + K^2) instead of O(n^2 K).
#
# The formulas are the ones in GIC_Calculation.jl (non-Huber versions).

using LinearAlgebra

# Value of criterion `crit` (larger is better). YHY and RSS are scalars for a
# vector response and q x q matrices for a matrix response.
function _gic_value(crit::Symbol, YHY, RSS, T::Integer, K::Integer, P::Integer)
    s2 = RSS / (T - K)
    fit = YHY / T
    if crit === :AIC
        return fit - K * s2 / T
    elseif crit === :AICc
        T - K - 2 <= 0 && return -999.0           # invalid model, as in Calculate_AIC_c
        return fit - (K + 1) * s2 / (T - K - 2)
    elseif crit === :SIC
        return fit - K * s2 / sqrt(T)
    elseif crit === :BIC
        return fit - K * s2 / T * log(T)
    elseif crit === :CAIC
        return fit - K * s2 / T * (1 + log(T))
    elseif crit === :GIC2
        return fit - K * s2 / T * P^(1 / 3)
    elseif crit === :GIC3
        return fit - 2 * K * s2 / T * log(P)
    elseif crit === :GIC4
        return fit - 2 * K * s2 / T * (log(P) + log(log(P)))
    elseif crit === :GIC5
        return fit - K * s2 / T * log(log(T)) * log(P)
    elseif crit === :GIC6
        return fit - K * s2 / T * log(T) * log(P)
    elseif crit === :EBIC
        return fit - K * s2 / T * (log(T) + log(K))
    else
        error("Unknown criterion: $crit")
    end
end

# Criterion functions that have a fast path; anything else uses the original loop.
_criterion_symbol(::Any) = nothing
for (crit, fname) in ((:AIC, :Calculate_AIC), (:AICc, :Calculate_AIC_c),
                      (:SIC, :Calculate_SIC), (:BIC, :Calculate_BIC),
                      (:CAIC, :Calculate_CAIC), (:GIC2, :Calculate_GIC2),
                      (:GIC3, :Calculate_GIC3), (:GIC4, :Calculate_GIC4),
                      (:GIC5, :Calculate_GIC5), (:GIC6, :Calculate_GIC6),
                      (:EBIC, :Calculate_EBIC))
    q = QuoteNode(crit)
    @eval _criterion_symbol(::typeof($fname)) = $q
end

# Inverse of the Gram matrix, via Cholesky when possible.
function _gram_inverse(G::AbstractMatrix)
    size(G, 1) == 0 && return zeros(0, 0)
    F = cholesky(Symmetric(G); check = false)
    return issuccess(F) ? inv(F) : inv(G)
end

mutable struct _SubsetFit
    S::Vector{Int}              # selected columns, in insertion order
    Xs::Matrix{Float64}         # X[:, S]
    M::Matrix{Float64}          # (Xs'Xs)^{-1}
    B::Matrix{Float64}          # Xs'Y   (K x q)
    YHY::Matrix{Float64}        # Y'HY   (q x q)
    accepted::Int               # moves since the last full refactorization
end

struct _SearchData
    X::Matrix{Float64}
    Yvec::Bool                  # response given as a vector
    XtY::Matrix{Float64}        # p x q
    colsq::Vector{Float64}      # ||x_j||^2
    YtY::Matrix{Float64}        # q x q
    crit::Symbol
    P::Int
end

function _SearchData(X, Y, crit::Symbol)
    Xf = Matrix{Float64}(X)
    Ym = Y isa AbstractVector ? reshape(Vector{Float64}(Y), :, 1) : Matrix{Float64}(Y)
    _SearchData(Xf, Y isa AbstractVector, Xf' * Ym, vec(sum(abs2, Xf; dims = 1)),
                Ym' * Ym, crit, size(Xf, 2))
end

function _refit(d::_SearchData, S::Vector{Int})
    Xs = d.X[:, S]
    M = Matrix{Float64}(_gram_inverse(Xs' * Xs))
    B = d.XtY[S, :]
    _SubsetFit(copy(S), Xs, M, B, Matrix{Float64}(B' * (M * B)), 0)
end

function _score(d::_SearchData, YHY::Matrix{Float64}, K::Int)
    RSS = d.YtY - YHY
    d.Yvec && return _gic_value(d.crit, YHY[1, 1], RSS[1, 1], size(d.X, 1), K, d.P)
    return _gic_value(d.crit, YHY, RSS, size(d.X, 1), K, d.P)
end
_score(d::_SearchData, f::_SubsetFit) = _score(d, f.YHY, length(f.S))

# Score of adding column z. Returns (value, cache), or (nothing, nothing) if
# x_z is (numerically) in the span of the current columns.
function _try_add(d::_SearchData, f::_SubsetFit, z::Int)
    xz = view(d.X, :, z)
    if isempty(f.S)
        Mu, s, e = Float64[], d.colsq[z], d.XtY[z, :]
    else
        u = f.Xs' * xz
        Mu = f.M * u
        s = d.colsq[z] - dot(u, Mu)
        e = d.XtY[z, :] - f.B' * Mu
    end
    s <= 1e-10 * max(d.colsq[z], eps()) && return (nothing, nothing)
    YHY = f.YHY + (e * e') ./ s
    return (_score(d, YHY, length(f.S) + 1), (Mu, s, YHY))
end

function _accept_add!(d::_SearchData, f::_SubsetFit, z::Int, cache)
    Mu, s, YHY = cache
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
    f.accepted += 1
    return f
end

# Score of removing the column at position j of f.S.
function _try_remove(d::_SearchData, f::_SubsetFit, j::Int)
    mjj = f.M[j, j]
    betaj = vec(f.M[j, :]' * f.B)
    YHY = f.YHY - (betaj * betaj') ./ mjj
    return (_score(d, YHY, length(f.S) - 1), YHY)
end

function _accept_remove!(d::_SearchData, f::_SubsetFit, j::Int, YHY)
    keep = [i for i in eachindex(f.S) if i != j]
    mj = f.M[keep, j]
    f.M = f.M[keep, keep] .- (mj * mj') ./ f.M[j, j]
    deleteat!(f.S, j)
    f.Xs = f.Xs[:, keep]
    f.B = f.B[keep, :]
    f.YHY = YHY
    f.accepted += 1
    return f
end

# Same search as the original loop in GIC_Model_Selection.jl: visit the columns
# in `visit` order; remove a column if present, add it if absent, and keep the
# move if the criterion increases. Selected sets are reported in the original
# order (new columns first).
function _GIC_Variable_Selection_fast(X, Y, Init_Columns, crit::Symbol, visit)
    n = size(X, 1)
    d = _SearchData(X, Y, crit)
    f = _refit(d, reverse(collect(Int, Init_Columns)))
    current = _score(d, f)
    inS = falses(size(X, 2))
    inS[f.S] .= true

    GIC_list = zeros(length(visit))
    GIC_coeff = [Int[] for _ in 1:length(visit)]
    for (w, z) in enumerate(visit)
        if inS[z]
            j = findfirst(==(z), f.S)
            value, cache = _try_remove(d, f, j)
            if tr(current) < tr(value)
                _accept_remove!(d, f, j, cache)
                inS[z] = false
                current = value
            end
        elseif length(f.S) + 1 < n           # keep T - K > 0
            value, cache = _try_add(d, f, z)
            if value !== nothing && tr(current) < tr(value)
                _accept_add!(d, f, z, cache)
                inS[z] = true
                current = value
            end
        end
        if f.accepted >= 200                 # refactorize now and then to avoid drift
            f = _refit(d, f.S)
            current = _score(d, f)
        end
        GIC_list[w] = tr(current)
        GIC_coeff[w] = reverse(f.S)
    end
    return (GIC_list, GIC_coeff)
end
