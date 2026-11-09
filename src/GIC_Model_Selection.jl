using Random
using LinearAlgebra

function GIC_Variable_Selection(
    X::AbstractMatrix, 
    Y::Union{AbstractVector, AbstractMatrix}, 
    Init_Columns::AbstractVector{Int}, 
    Calculate_GIC, 
    Calculate_GIC_short;
    Huber::Bool = false,
    Nsim::Int64 = 2,
    fast_off::Bool = false   # true: always use the original (slow) loop
)
    # --- Input Validation ---
    m, n = size(X)
    @assert length(Init_Columns) ≤ n "Initial columns exceed design matrix size."
    @assert m == size(Y, 1) "Sample size mismatch between X and Y."

    # --- Initialization ---
    shuffled_sets = [shuffle(1:n) for _ in 1:Nsim]   # List of Nsim shuffled vectors
    repeated_list = vcat(shuffled_sets...) 

    # Fast path (GIC_Fast.jl): rank-one updates instead of refitting every
    # candidate. Used for the built-in criteria without Huber; same search.
    crit = _criterion_symbol(Calculate_GIC)
    if !Huber && crit !== nothing && !fast_off
        return _GIC_Variable_Selection_fast(X, Y, Init_Columns, crit, repeated_list)
    end
    # sets = collect(1:n)  # All feature indices
    # repeated_list = repeat(sets, Nsim)  # Feature sequence for iteration

    # Initial GIC calculation and inverse covariance
    GIC_coef_sets = Init_Columns
    GIC_c, M_inv = Calculate_GIC(Y, X[:, GIC_coef_sets], Huber)
    current_X = X[:, GIC_coef_sets]

    # --- Output Storage ---
    GIC_list = zeros(length(repeated_list))
    GIC_coeff = [Int[] for _ in 1:length(repeated_list)]
    w = 1  # Iteration counter

    # --- Main Iteration Loop ---

    for z in repeated_list
        if z in GIC_coef_sets  
            # Case 1: Test removing feature `z`
            index = findfirst(==(z), GIC_coef_sets)
            GIC_coef_sets_temp = deleteat!(copy(GIC_coef_sets), index)
            X_subsets = X[:, GIC_coef_sets_temp]

            # Block matrix update for inverse covariance (efficient removal)
            # A_hat = M_inv[setdiff(1:end, index), setdiff(1:end, index)]
            # B_hat = M_inv[setdiff(1:end, index), index]
            # C_hat = M_inv[index, setdiff(1:end, index)]'
            # D_hat = only(M_inv[index, index])
            # A_inv = A_hat - ((B_hat / D_hat) * C_hat)  # Schur complement

            # GIC evaluation after removal
            #GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)
            GIC_i, A_inv = Calculate_GIC(Y, X_subsets, Huber)

            if tr(GIC_c) < tr(GIC_i)  # Keep change if GIC improves
                GIC_c = GIC_i
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            end
            GIC_list[w] = tr(GIC_c)
            GIC_coeff[w] = copy(GIC_coef_sets)

        elseif z ∉ GIC_coef_sets  
            # Case 2: Test adding feature `z`
            GIC_coef_sets_temp = prepend!(copy(GIC_coef_sets), z)
            X_subsets = X[:, GIC_coef_sets_temp]
            index = findfirst(==(z), GIC_coef_sets_temp)

            # Block matrix update for inverse covariance (efficient addition)
            # Xsquare = X_subsets' * X_subsets
            # A_hat = Xsquare[setdiff(1:end, index), setdiff(1:end, index)]
            # B_hat = Xsquare[setdiff(1:end, index), index]
            # C_hat = Xsquare[index, setdiff(1:end, index)]'
            # D_hat = only(Xsquare[index, index])

            # Sherman-Morrison-Woodbury formula
            # topleft = M_inv + M_inv * B_hat * inv(D_hat - C_hat * M_inv * B_hat) * C_hat * M_inv
            # topright = -M_inv * B_hat * inv(D_hat - C_hat * M_inv * B_hat)
            # bottomleft = -inv(D_hat - C_hat * M_inv * B_hat) * C_hat * M_inv
            # bottomright = inv(D_hat - C_hat * M_inv * B_hat)
            # A_inv = [topleft topright; bottomleft bottomright]

            # GIC evaluation after addition
            #GIC_i = Calculate_GIC_short(Y, X_subsets, A_inv)
            GIC_i, A_inv = Calculate_GIC(Y, X_subsets, Huber)

            if tr(GIC_c) < tr(GIC_i)  # Keep change if GIC improves
                GIC_c = GIC_i
                GIC_coef_sets = GIC_coef_sets_temp
                M_inv = A_inv
                current_X = X_subsets 
            end
            GIC_list[w] = tr(GIC_c)
            GIC_coeff[w] = copy(GIC_coef_sets)
        end
        w += 1
    end

    return (GIC_list, GIC_coeff)
end