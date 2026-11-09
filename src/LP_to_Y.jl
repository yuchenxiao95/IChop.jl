##############################################################################
#  LP_to_Y.jl  --  simulate responses from a linear predictor eta = X * beta
#
#  Family              Link                        Extra arguments
#  ------------------  --------------------------  ----------------------------
#  Normal              identity                    std (default 1)
#  Bernoulli           logit
#  Binomial            logit                       n_trials (required)
#  Poisson             log
#  Gamma               log, mean exp(eta)          shape (default 1)
#  Exponential         log, mean exp(eta)
#  InverseGaussian     log, mean exp(eta)          shape (lambda, default 1)
#  NegativeBinomial    log, mean exp(eta)          theta (size, default 1)
#  Geometric           log, mean exp(eta)
#  MultivariateNormal  identity (n x q eta)        cov_matrix (q x q, required)
#  Multinomial         baseline-category logit     n_categories (required);
#                      (last category = baseline)  beta is p x (n_categories-1)
#
#  Log-link linear predictors are clamped to [-30, 30] to avoid overflow.
##############################################################################
using LinearAlgebra
using Distributions

const _LOG_LINK_FAMILIES = ("Poisson", "Gamma", "Exponential", "InverseGaussian",
                            "NegativeBinomial", "Geometric")

_lp_softmax(z) = (e = exp.(z .- maximum(z)); e ./ sum(e))
_inv_logit(x) = 1 / (1 + exp(-x))

function LP_to_Y(X::AbstractMatrix, true_beta::Union{AbstractVector, AbstractMatrix},
                 family::String = "Normal";
                 n_trials::Union{Nothing, Integer} = nothing,
                 std::Union{Nothing, Real} = nothing,
                 shape::Union{Nothing, Real} = nothing,
                 theta::Union{Nothing, Real} = nothing,
                 cov_matrix::Union{Nothing, AbstractMatrix} = nothing,
                 n_categories::Union{Nothing, Integer} = nothing)

    eta = X * true_beta

    if family in _LOG_LINK_FAMILIES
        size(eta, 2) == 1 || error("Family $family expects a vector beta.")
        mu = exp.(clamp.(vec(eta), -30.0, 30.0))
        if family == "Poisson"
            return rand.(Poisson.(mu))
        elseif family == "Gamma"
            a = isnothing(shape) ? 1.0 : Float64(shape)
            return rand.(Gamma.(a, mu ./ a))                 # mean mu
        elseif family == "Exponential"
            return rand.(Exponential.(mu))                   # mean mu
        elseif family == "InverseGaussian"
            lambda = isnothing(shape) ? 1.0 : Float64(shape)
            return rand.(InverseGaussian.(mu, lambda))       # mean mu
        elseif family == "NegativeBinomial"
            r = isnothing(theta) ? 1.0 : Float64(theta)
            return rand.(NegativeBinomial.(r, r ./ (r .+ mu)))  # mean mu
        else # Geometric: number of failures, mean (1 - p) / p = mu
            return rand.(Geometric.(1 ./ (1 .+ mu)))
        end

    elseif family == "Normal"
        s = isnothing(std) ? 1.0 : Float64(std)
        return eta .+ s .* randn(size(eta)...)

    elseif family == "Bernoulli"
        return Float64.(rand.(Bernoulli.(_inv_logit.(vec(eta)))))

    elseif family == "Binomial"
        isnothing(n_trials) && error("n_trials must be specified for Binomial family")
        return rand.(Binomial.(n_trials, _inv_logit.(vec(eta))))

    elseif family == "MultivariateNormal"
        isnothing(cov_matrix) && error("cov_matrix must be specified for MultivariateNormal")
        isposdef(cov_matrix) || error("Covariance matrix must be positive definite")
        size(cov_matrix, 1) == size(eta, 2) ||
            error("cov_matrix must be q x q, where q = number of columns of true_beta")
        L = cholesky(Symmetric(Matrix{Float64}(cov_matrix))).L
        return eta .+ randn(size(eta)...) * L'               # rows ~ N(eta_i, Sigma)

    elseif family == "Multinomial"
        isnothing(n_categories) && error("n_categories must be specified for Multinomial")
        n_categories < 2 && error("n_categories must be ≥ 2")
        size(true_beta, 2) == n_categories - 1 ||
            error("true_beta must have size (n_features × (n_categories-1))")
        return [rand(Categorical(_lp_softmax([row; 0.0]))) for row in eachrow(eta)]

    else
        error("Unsupported family: $family")
    end
end
