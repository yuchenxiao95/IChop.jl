##############################################################################
#  GIC_Model_Selection_Boltzmann.jl
#
#  Stochastic (Boltzmann / simulated-annealing style) version of the GIC
#  coordinate search. Requires penalty_selection.jl and GIC_Model_Selection.jl
#  (for the incremental least-squares engine) to be loaded first.
##############################################################################
using Random
using LinearAlgebra

_logistic(x::Real) = x >= 0 ? 1 / (1 + exp(-x)) : (ex = exp(x); ex / (1 + ex))

"""
    GIC_Variable_Selection_Boltzmann(X, Y, Init_Columns, Calculate_GIC, Calculate_GIC_short;
                                     T = 0.2, Nsim = 2, k = typemax(Int), gamma = 0.6)

Like `GIC_Variable_Selection`, but a proposed add/remove move is accepted with
probability `logistic((GIC_new - GIC_current) / T)` instead of only when the
criterion improves. Small `T` gives an almost greedy search; large `T` explores
more. The columns are shuffled once and visited `Nsim` times. Moves that would
give a model with `k` or more columns are skipped (no cap by default), and
models never exceed n - 1 columns.

Returns `(GIC_list, GIC_coeff)` as `GIC_Variable_Selection` does.
"""
function GIC_Variable_Selection_Boltzmann(
    X::AbstractMatrix,
    Y::Union{AbstractVector, AbstractMatrix},
    Init_Columns::AbstractVector{<:Integer},
    Calculate_GIC,
    Calculate_GIC_short;
    T::Real = 0.2,
    Nsim::Integer = 2,
    k::Integer = typemax(Int),
    gamma::Real = 0.6
)
    m, p = size(X)
    kcap = min(k, m)          # models have at most min(k, n) - 1 variables
    @assert m == size(Y, 1) "Sample size mismatch between X and Y."
    T > 0 || error("Temperature T must be positive.")

    visit = repeat(shuffle(collect(1:p)), Nsim)
    accept(new, old) = rand() < _logistic((tr(new) - tr(old)) / T)

    GIC_list = zeros(length(visit))
    GIC_coeff = Vector{Any}(undef, length(visit))    # Vector{Any} -> plain R list

    crit = _criterion_symbol(Calculate_GIC)
    if crit === nothing                          # user-supplied criterion
        S = collect(Int, Init_Columns)
        current, _ = Calculate_GIC(Y, X[:, S], p)
        for (w, z) in enumerate(visit)
            idx = findfirst(==(z), S)
            cand = idx === nothing ? (length(S) + 1 < kcap ? vcat(z, S) : nothing) :
                                     deleteat!(copy(S), idx)
            if cand !== nothing
                value, _ = Calculate_GIC(Y, X[:, cand], p)
                accept(value, current) && ((current, S) = (value, cand))
            end
            GIC_list[w] = tr(current)
            GIC_coeff[w] = copy(S)
        end
        return (GIC_list, GIC_coeff)
    end

    d = _SearchData(X, Y, crit, p, gamma)
    f = _refit(d, reverse(collect(Int, Init_Columns)))
    current = _score(d, f)
    inS = falses(p)
    inS[f.S] .= true

    for (w, z) in enumerate(visit)
        if inS[z]
            j = findfirst(==(z), f.S)
            value, cache = _try_remove(d, f, j)
            if accept(value, current)
                _accept_remove!(d, f, j, cache)
                inS[z] = false
                current = value
            end
        elseif length(f.S) + 1 < kcap
            value, cache = _try_add(d, f, z)
            if value !== nothing && accept(value, current)
                _accept_add!(d, f, z, cache)
                inS[z] = true
                current = value
            end
        end
        if f.accepted >= 200
            f = _refit(d, f.S)
            current = _score(d, f)
        end
        GIC_list[w] = tr(current)
        GIC_coeff[w] = _report(f)
    end
    return (GIC_list, GIC_coeff)
end

# Entry point for the R wrapper; the criteria are passed by name (see
# _GIC_Variable_Selection_flat in GIC_Model_Selection.jl).
_GIC_Variable_Selection_Boltzmann_byname(X, Y, Init_Columns, Calculate_GIC::AbstractString,
                                         Calculate_GIC_short::AbstractString; kwargs...) =
    GIC_Variable_Selection_Boltzmann(X, Y, Init_Columns, _julia_function(Calculate_GIC),
                                     _julia_function(Calculate_GIC_short); kwargs...)
