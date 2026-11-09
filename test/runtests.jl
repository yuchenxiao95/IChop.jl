using GICVariableSelection
using Random, LinearAlgebra
using Test

# Final model of a search
last_model(fit) = sort(fit[2][end])

@testset "Linear model" begin
    Random.seed!(1)
    n, p = 1000, 100
    X = randn(n, p)
    truth = sort(randperm(p)[1:5])
    beta = zeros(p); beta[truth] .= 2
    Y = LP_to_Y(X, beta, "Normal"; std = 2.0)

    fit = GIC_Variable_Selection(X, Y, collect(1:p), Calculate_GIC6, Calculate_GIC6_short, p;
                                 Nsim = 3)
    @test length(fit[1]) == 3p
    @test length(fit[2]) == 3p
    @test last_model(fit) == truth

    # The fast path gives the same search as the generic one (a wrapped criterion)
    Random.seed!(2); fast = GIC_Variable_Selection(X, Y, 1:10, Calculate_BIC, Calculate_BIC_short, p)
    slow_crit(Y, Xs, P) = Calculate_BIC(Y, Xs, P)
    Random.seed!(2); slow = GIC_Variable_Selection(X, Y, 1:10, slow_crit, nothing, p)
    @test sort(fast[2][end]) == sort(slow[2][end])
    @test fast[1] ≈ slow[1]
end

@testset "Model-size cap k" begin
    Random.seed!(3)
    n, p = 500, 60
    X = randn(n, p)
    Y = X[:, 1:20] * ones(20) + randn(n)
    fit = GIC_Variable_Selection(X, Y, Int[], Calculate_AIC, Calculate_AIC_short, 10; Nsim = 2)
    @test maximum(length, fit[2]) <= 9
end

@testset "Multivariate normal" begin
    Random.seed!(4)
    n, p, m = 1000, 100, 3
    X = randn(n, p)
    B = zeros(p, m)
    truth = Int[]
    for i in 1:m
        cols = randperm(p)[1:4]
        B[cols, i] .= 3
        append!(truth, cols)
    end
    Y = LP_to_Y(X, B, "MultivariateNormal"; cov_matrix = Matrix(1.0I, m, m))
    fit = GIC_Variable_Selection(X, Y, collect(1:p), Calculate_BIC, Calculate_BIC_short, p;
                                 Nsim = 3)
    @test isempty(setdiff(unique(truth), last_model(fit)))
end

@testset "Poisson and multinomial" begin
    Random.seed!(5)
    n, p = 2000, 50
    X = randn(n, p)
    truth = [3, 17, 42]
    beta = zeros(p); beta[truth] = [1.0, -1.0, 0.8]

    y = LP_to_Y(X, beta, "Poisson")
    fit = GIC_Variable_Selection(X, Y_to_LP(y, "Poisson"), collect(1:10),
                                 Calculate_GIC6, Calculate_GIC6_short, p; Nsim = 3)
    @test last_model(fit) == truth
    b = Beta_estimate(y, X[:, truth]; family = "Poisson")
    @test length(b) == 4
    @test isapprox(b[2:end], beta[truth]; atol = 0.1)

    B = zeros(p, 2); B[truth, 1] .= 1.5; B[truth, 2] = [-1.5, 1.5, -1.5]
    y_mn = LP_to_Y(X, B, "Multinomial"; n_categories = 3)
    @test sort(unique(y_mn)) == [1, 2, 3]
    lp = Y_to_LP(y_mn, "Multinomial", n_categories = 3)
    @test size(lp) == (n, 2)
end
