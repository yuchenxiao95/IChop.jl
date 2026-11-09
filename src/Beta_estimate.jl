using LinearAlgebra, Statistics
using DataFrames, GLM
import GLM: LogitLink, LogLink, IdentityLink, InverseLink
using Flux # Add Flux dependency for multinomial regression
using Flux: onehotbatch, onecold, crossentropy, DataLoader, logitcrossentropy # Specific Flux imports
using Optimisers # For Optimisers.setup and Optimisers.update

"""
    Beta_estimate(Y, X; family=:auto, link=:auto,
                  n_trials=nothing, add_intercept=true)

Estimate regression coefficients for many response types.

If `add_intercept = true` (default) a leading column of 1s is prepended to
`X` and the returned vector/matrix includes β₀ as the first row.
"""
# ─────────────────────────────────────────────────────────────────────────────
# 3 · Beta_estimate  — fit coefficients (includes intercept)
# ─────────────────────────────────────────────────────────────────────────────
"""
    Beta_estimate(Y, X; family=:auto, …)

Return β̂.  If `add_intercept=true` (default) the first row is the intercept.
For `:Multinomial`, returns a `(p+1) × (K-1)` matrix if `add_intercept=true`,
or `p × (K-1)` matrix if `add_intercept=false`. The coefficients are
relative to the last category as the reference class.
"""
function Beta_estimate(
        Y, X;
        family        ::Union{Symbol,String} = :auto,
        link          ::Union{Link,Symbol}   = :auto,
        n_trials      ::Union{Nothing,Int}   = nothing,
        n_categories  ::Union{Nothing,Int}   = nothing,
        add_intercept ::Bool                 = true)

    # Ensure X is Float32 for Flux compatibility if it's used.
    # X_input_to_function retains the (samples x features) structure for initial checks.
    X_input_to_function = Float32.(X)

    if isa(Y, AbstractMatrix) && size(Y,2)==1; Y = vec(Y) end

    family = family === :auto ? begin
        if isa(Y, AbstractMatrix)
            :MultivariateNormal
        elseif eltype(Y)<:Bool || all(y->y in (0,1), Y)
            :Bernoulli
        elseif eltype(Y)<:Integer && all(Y .>= 0)
            length(unique(Y)) > 2 ? :Multinomial :
            isnothing(n_trials) ? :Poisson : :Binomial
        else                              :Linear
        end
    end : Symbol(family)

    p_orig = size(X_input_to_function, 2) # Number of original features (excluding potential intercept)

    # 1 · Linear (uses GLM.jl logic)
    if family == :Linear
        X_glm = add_intercept ? hcat(ones(size(X_input_to_function,1)), X_input_to_function) : copy(X_input_to_function)
        return Vector((X_glm'X_glm) \ (X_glm'Y))
    end

    # 2 · Multivariate Normal (uses GLM.jl logic)
    if family == :MultivariateNormal
        X_glm = add_intercept ? hcat(ones(size(X_input_to_function,1)), X_input_to_function) : copy(X_input_to_function)
        return Matrix((X_glm'X_glm) \ (X_glm'Y))
    end

    # -- Multinomial (TRUE Multinomial Logistic Regression with Softmax via Flux) --
    if family == :Multinomial
        # Determine K (number of categories) and the actual categories
        # More robust category determination
        unique_Y_values = sort(unique(Y))
        if isnothing(n_categories)
            K = length(unique_Y_values)
            categories = unique_Y_values
        else
            K = n_categories
            # Validate if all Y values are within the expected range for n_categories
            # Or assume categories are 0:(K-1) or 1:K if not explicitly provided
            if minimum(Y) < (maximum(Y) - K + 1) || maximum(Y) > (minimum(Y) + K - 1)
                 @warn "n_categories is provided but does not match the range of unique Y values. Using unique Y values to define categories."
                 K = length(unique_Y_values)
                 categories = unique_Y_values
            else
                # This assumes categories are contiguous if n_categories is given without explicit list
                categories = (minimum(Y) == 0) ? (0:(K-1)) : (1:K)
            end
        end

        # Ensure that categories given to onehotbatch matches the actual values in Y
        if !issubset(unique_Y_values, categories)
             error("The provided Y values contain categories not present in the defined range (either inferred or from n_categories).")
        end

        # One-hot encode Y for Flux.
        Y_onehot = onehotbatch(Y, categories)

        # Prepare X for Flux: Flux expects (features x samples) format.
        X_for_flux_model = permutedims(X_input_to_function) # Now (p_orig x samples)

        input_features_for_dense = p_orig # Number of features for the Dense layer input

        # --- Define the Multinomial Logistic Regression Model using Flux ---
        # The `bias` argument in Dense corresponds to the intercept.
        model = Chain(
            Dense(input_features_for_dense, K, bias=add_intercept),
            softmax # Apply softmax to get probabilities
        )

        # Loss function for multinomial classification (numerically stable)
        loss_fn_for_grad(m, x, y) = Flux.Losses.logitcrossentropy(m.layers[1](x), y)

        # Optimizer setup
        opt = Optimisers.Adam(1e-3) # A common learning rate
        state = Optimisers.setup(opt, model)

        # DataLoader for training
        train_loader = DataLoader((X_for_flux_model, Y_onehot), batchsize = 32, shuffle = true) # Smaller batch size for potentially better generalization

        # --- Training Loop ---
        num_epochs = 200 # Increased epochs for better convergence
        for epoch in 1:num_epochs
            for (x_batch, y_batch) in train_loader
                grads = gradient((m) -> loss_fn_for_grad(m, x_batch, y_batch), model)
                state, model = Optimisers.update(state, model, grads[1])
            end
            # Optional: Print loss every few epochs to monitor training
            # if epoch % 20 == 0
            #     current_loss = sum(loss_fn_for_grad(model, x, y) for (x, y) in train_loader)
            #     println("Epoch $epoch, Loss: $current_loss")
            # end
        end

        # --- Extract Coefficients (β) from the Flux model ---
        # model[1] refers to the Dense layer in the Chain
        W_flux = model[1].weight # K x p_orig (features)
        b_flux = model[1].bias   # K (empty if bias=false)

        # The coefficients are typically presented relative to a reference class.
        # A common choice is the last category.
        # The last category in `categories` will correspond to the K-th row/element of W_flux and b_flux.
        # Assuming `categories` is sorted.

        coef_rows = add_intercept ? (p_orig + 1) : p_orig
        β_multinomial = zeros(Float64, coef_rows, K - 1)

        # The categories array is 1-indexed in Julia, but the actual values can be 0 or more.
        # W_flux and b_flux are indexed from 1 to K.
        # The last category in the sorted `categories` list is the reference.
        # Its index in the Flux parameters will be K.
        reference_category_index_in_flux = K # This refers to the K-th row/element of W_flux/b_flux

        if !add_intercept # No intercept in the model
            W_ref = W_flux[reference_category_index_in_flux, :] # Weights for reference class
            for k_idx in 1:(K-1) # Iterate through non-reference classes
                W_k = W_flux[k_idx, :]
                β_multinomial[1:p_orig, k_idx] = W_k .- W_ref # p_orig features
            end
        else # Intercept is included
            W_ref = W_flux[reference_category_index_in_flux, :] # Weights for reference class
            b_ref = b_flux[reference_category_index_in_flux]    # Bias for reference class

            for k_idx in 1:(K-1) # Iterate through non-reference classes
                W_k = W_flux[k_idx, :]
                b_k = b_flux[k_idx]

                # Intercept coefficient (first row of β_multinomial)
                β_multinomial[1, k_idx] = b_k - b_ref

                # Feature coefficients (remaining rows)
                β_multinomial[2:end, k_idx] = W_k .- W_ref # p_orig features
            end
        end

        return β_multinomial
    end

    # 4 · scalar GLMs (Original GLM.jl based code)
    # These parts of the code still use DataFrames and GLM.jl directly
    X_glm = add_intercept ? hcat(ones(size(X_input_to_function,1)), X_input_to_function) : copy(X_input_to_function)
    p_glm = size(X_glm,2) # Features for GLM.jl, including intercept

    fam = family == :Bernoulli ? Binomial() :
           family == :Binomial  ? Binomial(isnothing(n_trials) ? 1 : n_trials) :
           family == :Poisson   ? Poisson() :
           family == :Gamma     ? Gamma() :
           family == :Exponential ? Gamma() :
           error("Unsupported family $family")

    link = link === :auto ? (
             fam isa Binomial ? LogitLink() :
             fam isa Poisson  ? LogLink()   :
             fam isa Gamma    ? InverseLink() : IdentityLink()
           ) : link

    # Create DataFrame for GLM.jl
    df  = DataFrame(X_glm, Symbol.("x",1:p_glm)); df.y = Y
    rhs = join(Symbol.("x",1:p_glm)," + ")
    fml = @eval @formula(y ~ $(Meta.parse(rhs)))
    mdl = glm(fml, df, fam, link)
    return coef(mdl) # includes intercept row
end