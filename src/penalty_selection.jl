##############################################################################
#  penalty_selection.jl
#
#  Information criteria for GIC-based variable selection.
#
#  Every criterion is a function of a few sufficient quantities of the
#  least-squares fit on the column subset X_S:
#
#      YHY = Y' H Y = b' M b,   b = X_S'Y,  M = (X_S'X_S)^{-1}
#      RSS = Y'Y - YHY,         s2 = RSS / (T - K)
#
#  so the n x n hat matrix H is never formed. The public functions keep the
#  original names and signatures:
#
#      Calculate_<IC>(Y, X[, P])                -> (value, Inverse)
#      Calculate_<IC>_short(Y, X, Inverse[, P]) -> value
#
#  and the search drivers in GIC_Model_Selection*.jl call the same core with
#  incrementally updated quantities.
##############################################################################
using LinearAlgebra

# log(binomial(P, K)) in Float64, without BigInt arithmetic.
function _logbinom(P::Integer, K::Integer)
    (K < 0 || K > P) && return -Inf
    K = min(K, P - K)
    s = 0.0
    @inbounds for i in 1:K
        s += log((P - K + i) / i)
    end
    return s
end

_softmax(z) = (e = exp.(z .- maximum(z)); e ./ sum(e))

# Criteria that need more than (YHY, RSS, K).
_needs_fitted(crit::Symbol) = crit === :AttIC
_needs_logdet(crit::Symbol) = crit === :CAICF

"""
    _gic_value(crit, YHY, RSS, T, K, P; Y, fitted, logdetG, gamma)

Value of criterion `crit` (larger is better). `YHY` and `RSS` are scalars for a
vector response and q x q matrices for a matrix response.
"""
function _gic_value(crit::Symbol, YHY, RSS, T::Integer, K::Integer, P::Integer;
                    Y = nothing, fitted = nothing, logdetG::Real = 0.0,
                    gamma::Real = 0.6)
    s2 = RSS / (T - K)
    fit = YHY / T
    if crit === :AIC
        return fit - 2 * K * s2 / T
    elseif crit === :AICc
        T - K - 2 <= 0 && return -999.0           # invalid model (as before)
        return fit - (K + 1) * s2 / (T - K - 2)
    elseif crit === :AttIC
        return Y' * _softmax(fitted) - K * s2 / sqrt(T)
    elseif crit === :SIC
        return fit - K * s2 / sqrt(T)
    elseif crit === :BIC
        return fit - K * s2 / T * log(T)
    elseif crit === :CAIC
        return fit - K * s2 / T * (1 + log(T))
    elseif crit === :CAICF
        s2 isa Number || error("CAICF is only defined for a vector response.")
        # log det(X'X / s2) = logdet(X'X) - K log(s2)
        return fit - K * s2 * (1 + log(T)) / T - s2 * (logdetG - K * log(s2)) / T
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
        return fit - K * s2 / T * (log(T) + gamma * _logbinom(P, K))
    else
        error("Unknown criterion: $crit")
    end
end

# Inverse and log-determinant of the Gram matrix, via Cholesky when possible.
function _gram_inverse(G::AbstractMatrix)
    size(G, 1) == 0 && return (zeros(0, 0), 0.0)
    F = cholesky(Symmetric(G); check = false)
    if issuccess(F)
        return (inv(F), logdet(F))
    end
    return (inv(G), first(logabsdet(G)))         # same behaviour as inv(X'X)
end

# Criterion from a design matrix (optionally with a precomputed inverse).
function _criterion_from_X(crit::Symbol, Y, X::AbstractMatrix, P::Integer;
                           Inverse = nothing, gamma::Real = 0.6)
    T, K = size(X)
    if Inverse === nothing
        Inverse, logdetG = _gram_inverse(X' * X)
    else
        logdetG = (_needs_logdet(crit) && K > 0) ? -first(logabsdet(Inverse)) : 0.0
    end
    b = X' * Y
    coef = Inverse * b
    fitted = X * coef
    resid = Y - fitted
    RSS = resid' * resid
    YHY = b' * coef
    value = _gic_value(crit, YHY, RSS, T, K, P; Y = Y, fitted = fitted,
                       logdetG = logdetG, gamma = gamma)
    return (value, Inverse)
end

const _CRITERIA = (
    (:AIC,   :Calculate_AIC),
    (:AICc,  :Calculate_AIC_c),
    (:AttIC, :Calculate_AttIC),
    (:SIC,   :Calculate_SIC),
    (:BIC,   :Calculate_BIC),
    (:CAIC,  :Calculate_CAIC),
    (:CAICF, :Calculate_CAICF),
    (:GIC2,  :Calculate_GIC2),
    (:GIC3,  :Calculate_GIC3),
    (:GIC4,  :Calculate_GIC4),
    (:GIC5,  :Calculate_GIC5),
    (:GIC6,  :Calculate_GIC6),
    (:EBIC,  :Calculate_EBIC),
)

const _YType = Union{AbstractVector, AbstractMatrix}

# Map a criterion function back to its symbol, so the search drivers can use
# the fast incremental path. Anything else falls back to calling the function.
_criterion_symbol(::Any) = nothing

for (crit, fname) in _CRITERIA
    short = Symbol(fname, :_short)
    q = QuoteNode(crit)
    @eval begin
        # Full form: returns (value, Inverse). P is accepted by every criterion
        # (and ignored by those that do not use it), so all have both arities.
        $fname(Y::_YType, X::AbstractMatrix, P::Integer; gamma::Real = 0.6) =
            _criterion_from_X($q, Y, X, P; gamma = gamma)
        $fname(Y::_YType, X::AbstractMatrix; gamma::Real = 0.6) =
            _criterion_from_X($q, Y, X, size(X, 2); gamma = gamma)

        # Short form: reuses a precomputed (X'X)^{-1}; returns the value only.
        $short(Y::_YType, X::AbstractMatrix, Inverse::AbstractMatrix, P::Integer;
               gamma::Real = 0.6) =
            first(_criterion_from_X($q, Y, X, P; Inverse = Inverse, gamma = gamma))
        $short(Y::_YType, X::AbstractMatrix, Inverse::AbstractMatrix;
               gamma::Real = 0.6) =
            first(_criterion_from_X($q, Y, X, size(X, 2); Inverse = Inverse, gamma = gamma))

        _criterion_symbol(::typeof($fname)) = $q
        _criterion_symbol(::typeof($short)) = $q
    end
end
