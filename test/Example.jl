import Pkg
Pkg.activate(".")

Pkg.rm("IChop")  # optional, to remove completely
Pkg.add(url="https://github.com/yuchenxiao95/IChop.jl")

using IChop, Plots, StatsBase, Distributions, DataFrames, LinearAlgebra, Random


Random.seed!(1234) 
N, P, k = 5000, 500, 10
rho = 0.0
true_columns = sort(sample(1:P, k, replace=false))
SNR = [0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]

mu = zeros(P)
cov = fill(rho, P, P)
for i in 1:P
cov[i, i] = 1.0
end
X = Matrix(rand(MvNormal(mu, cov), N)')

# # Simulate multicolinearity
# # Choose 5 of them to be replaced
# num_replace = 3
# replace_columns = sample(true_columns, num_replace; replace = false)
# other_columns = setdiff(1:P, true_columns)

# used_mix_cols = Vector{Vector{Int}}()  # This will be your list of lists

# used_cols = Set{Int}()  # Flat set of used columns to avoid reuse

# for col in replace_columns
#     available = setdiff(setdiff(other_columns, [col]), collect(used_cols))

#     if length(available) < 2
#         @warn "Not enough columns left to mix for column $col"
#         continue
#     end

#     mix = sample(available, rand(2:4); replace = false)
#     coeffs = 3 * rand(length(mix))

#     X[:, col] = X[:, mix] * coeffs

#     push!(used_mix_cols, mix)
#     union!(used_cols, mix)
# end


true_beta = zeros(P)
true_beta[true_columns] .= 3
variance = (true_beta' * cov * true_beta) / SNR[4]
std = sqrt(variance)

Y = LP_to_Y(X, true_beta, family="Normal", std=std)


#------------------------------------
# Get dimensions
T, K = size(true_X, 1), size(true_X, 2)
# Compute inverse and hat matrix
Inverse = inv(true_X' * true_X)
Hat_matrix = true_X * Inverse * true_X'
est_Y = Hat_matrix * Y
est_MSE = mean((est_Y .- Y).^2)
# Compute residuals and sample variance
residuals = Y - Hat_matrix * Y
sample_variance = (residuals' * residuals) / (T-K)
# Compute ICOMP
ICOMP = (Y' * Hat_matrix * Y) / T - (K * sample_variance) / sqrt(T) -
(K * log(abs(tr(sample_variance * Inverse) / K)) - logabsdet(sample_variance *Inverse)[1])

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Get dimensions
T, K = size(est_X , 1), size(est_X , 2)
# Compute inverse and hat matrix
Inverse = inv(est_X' * est_X)
Hat_matrix = est_X  * Inverse * est_X'
est_Y = Hat_matrix * Y
est_MSE = mean((est_Y .- Y).^2)
# Compute residuals and sample variance
residuals = Y - Hat_matrix * Y
sample_variance = (residuals' * residuals) / (T-K)
# Compute ICOMP
ICOMP = (Y' * Hat_matrix * Y) / T - (K * sample_variance) / sqrt(T) -
(K * log(abs(tr(sample_variance * Inverse) / K)) - logabsdet(sample_variance *Inverse)[1])
#-----------------------------------------------------------------------------------------

true_X = X[:,true_columns]
length(true_columns)*log(tr(inv(X[:,true_columns]' * X[:,true_columns]))/length(true_columns)) - log(det(inv(X[:,true_columns]' * X[:,true_columns])))



est_X =  X[:,tmp[2][end]]
length(tmp[2][end])*log(tr(inv( X[:,tmp[2][end]]' * X[:,tmp[2][end]] ))/length(tmp[2][end])) - log(det(inv(X[:,tmp[2][end]]' * X[:,tmp[2][end]])))


#init_cols  = collect(1:P)
init_cols = sort(sample(1:P, Int64(floor(P/2)), replace=false))
@time begin 
tmp = GIC_Variable_Selection(X, Y, init_cols, Calculate_SIC, Calculate_SIC_short, Nsim=6)
end

tmpp = DataFrame(A = tmp[1], 
                    B = tmp[2])
Plots.plot(tmp[1])
#random_index_true_columns = unique(reduce(vcat, multi_beta_true_columns)) 
print(setdiff(tmp[2][end],true_columns))
print(setdiff(true_columns, tmp[2][end]))




estimate_beta = zeros(P)
IC, Inverse = Calculate_ICOMP(Y, X[:,tmp[2][end]])
################
# U, S, V = svd(X[:,tmp[2][end]])
# # Invert the squared singular values
# S_inv2 = Diagonal(1 ./ (S .^ 2))
# # Compute (X'X)^-1 from SVD: V * S^-2 * V'
# Inverse = V * S_inv2 * V'
######################
print("ICOMP")
Beta_estimate(Y, X[:,tmp[2][end]], Inverse)
estimate_beta[tmp[2][end]] .= Beta_estimate(Y, X[:,tmp[2][end]], Inverse)
estimate_beta[true_columns]
sum((estimate_beta.- true_beta) .* (estimate_beta.- true_beta)) / P



estimate_beta = zeros(P)
IC, Inverse = Calculate_ICOMP(Y, X[:,true_columns])
print("True")
Beta_estimate(Y, X[:,true_columns], Inverse)
estimate_beta[true_columns] .= Beta_estimate(Y, X[:,true_columns], Inverse)
sum((estimate_beta .- true_beta) .* (estimate_beta .- true_beta)) / P




estimate_beta = zeros(P)
IC, Inverse = Calculate_ICOMP(Y, X)
print("Full")
Beta_estimate(Y, X, Inverse)
estimate_beta .= Beta_estimate(Y, X, Inverse)
estimate_beta[true_columns]
sum((estimate_beta .- true_beta) .* (estimate_beta .- true_beta)) / P


