##############################################################################
#  Beta_estimate.jl  --  coefficient estimates for many response types
#
#  Uses GLM.jl's matrix interface directly (no DataFrame / @formula), with the
#  same families and links as LP_to_Y.jl.
##############################################################################
using LinearAlgebra, Statistics
using GLM
import GLM: Link, LogitLink, LogLink, IdentityLink, InverseLink

# Negative binomial with unknown size theta: alternate a GLM fit at fixed theta
# with the Pearson moment equation  sum (y - mu)^2 / (mu + mu^2/theta) = n - p.
function _negbin_moment(X, y, link; maxiter::Int = 25, tol::Real = 1e-6)
    n, p = size(X)
    mdl = glm(X, y, Poisson(), link)
    theta = 1.0
    for _ in 1:maxiter
        mu = predict(mdl)
        pearson(t) = sum((y .- mu) .^ 2 ./ (mu .+ mu .^ 2 ./ t)) - (n - p)
        lo, hi = 1e-4, 1e6
        if pearson(hi) > 0             # overdispersed: root exists
            pearson(lo) > 0 && (lo = hi)
            for _ in 1:100
                mid = sqrt(lo * hi)
                pearson(mid) > 0 ? (hi = mid) : (lo = mid)
                hi / lo < 1 + 1e-8 && break
            end
            new_theta = sqrt(lo * hi)
        else
            new_theta = hi              # no overdispersion: essentially Poisson
        end
        mdl = glm(X, y, NegativeBinomial(new_theta), link)
        abs(new_theta - theta) <= tol * theta && (theta = new_theta; break)
        theta = new_theta
    end
    return mdl
end

"""
    Beta_estimate(Y, X; family = :auto, link = :auto, n_trials = nothing,
                  theta = nothing, add_intercept = true)

Estimate regression coefficients.

Families: `:Linear` / `:Normal`, `:MultivariateNormal`, `:Bernoulli`,
`:Binomial` (counts out of `n_trials`), `:Poisson`, `:Gamma`, `:Exponential`,
`:InverseGaussian`, `:NegativeBinomial` (size `theta`; estimated by the
Pearson moment equation if `nothing`), `:Geometric`, `:Multinomial` (one-vs-rest logistic, last class as
reference). With `family = :auto` the family is guessed from `Y`.

Links default to the ones used by `LP_to_Y`: identity (Normal), logit
(Bernoulli, Binomial, Multinomial) and log (all others). If
`add_intercept = true`, a column of 1s is prepended and the first coefficient
is the intercept.
"""
function Beta_estimate(
        Y, X;
        family        ::Union{String,Symbol} = :auto,
        link          ::Union{Link,Symbol,String} = :auto,
        n_trials      ::Union{Nothing,Int}   = nothing,
        theta         ::Union{Nothing,Real}  = nothing,
        add_intercept ::Bool                 = true)

    X = Matrix{Float64}(X)
    add_intercept && (X = hcat(ones(size(X, 1)), X))
    p = size(X, 2)

    if Y isa AbstractMatrix && size(Y, 2) == 1
        Y = vec(Y)
    end

    # -- family detection -----------------------------------------------------
    family isa String && (family = Symbol(family))
    link isa String && (link = Symbol(link))
    if family === :auto
        family = if Y isa AbstractMatrix
                     :MultivariateNormal
                 elseif eltype(Y) <: Bool || all(y -> y in (0, 1), Y)
                     :Bernoulli
                 elseif all(y -> y >= 0 && isinteger(y), Y)
                     isnothing(n_trials) ? :Poisson : :Binomial
                 else
                     :Linear
                 end
    else
        family = Symbol(family)
    end
    family === :Normal && (family = :Linear)
    # A link given by name, e.g. :LogLink or :IdentityLink
    if link isa Symbol && link !== :auto
        link = getfield(GLM, link)()
    end

    # -- least squares --------------------------------------------------------
    if family === :Linear
        return Vector(X \ Vector{Float64}(Y))
    elseif family === :MultivariateNormal
        return Matrix(X \ Matrix{Float64}(Y))
    end

    # -- multinomial: one-vs-rest logistic ------------------------------------
    if family === :Multinomial
        classes = sort!(collect(unique(Y)))
        beta_hat = zeros(p, length(classes) - 1)
        for (j, c) in enumerate(classes[1:end-1])
            mdl = glm(X, Float64.(Y .== c), Binomial(), LogitLink())
            beta_hat[:, j] = coef(mdl)
        end
        return beta_hat
    end

    Y isa AbstractMatrix && error("Family $family expects a vector response; got a matrix.")
    y = Vector{Float64}(Y)

    default_link(l) = link === :auto ? l : link

    mdl = if family === :Bernoulli
        glm(X, y, Binomial(), default_link(LogitLink()))
    elseif family === :Binomial
        isnothing(n_trials) && error("n_trials must be specified for the Binomial family.")
        glm(X, y ./ n_trials, Binomial(), default_link(LogitLink());
            wts = fill(Float64(n_trials), length(y)))
    elseif family === :Poisson
        glm(X, y, Poisson(), default_link(LogLink()))
    elseif family === :Gamma || family === :Exponential
        glm(X, y, Gamma(), default_link(LogLink()))
    elseif family === :InverseGaussian
        glm(X, y, InverseGaussian(), default_link(LogLink()))
    elseif family === :Geometric
        glm(X, y, Geometric(), default_link(LogLink()))
    elseif family === :NegativeBinomial
        isnothing(theta) ? _negbin_moment(X, y, default_link(LogLink())) :
                           glm(X, y, NegativeBinomial(Float64(theta)), default_link(LogLink()))
    else
        error("Unsupported family token: $family")
    end
    return coef(mdl)
end
