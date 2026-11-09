module IChop

# The source files are shared with the R package IChop
# (inst/julia/), so keep them identical when changing either.

using LinearAlgebra
using Statistics
using Random
using Distributions
using GLM

include("penalty_selection.jl")
include("GIC_Model_Selection.jl")
include("GIC_Model_Selection_Boltzmann.jl")
include("LP_to_Y.jl")
include("Y_to_LP.jl")
include("Beta_estimate.jl")

export Calculate_AIC, Calculate_AIC_short,
       Calculate_AIC_c, Calculate_AIC_c_short,
       Calculate_AttIC, Calculate_AttIC_short,
       Calculate_SIC, Calculate_SIC_short,
       Calculate_BIC, Calculate_BIC_short,
       Calculate_CAIC, Calculate_CAIC_short,
       Calculate_CAICF, Calculate_CAICF_short,
       Calculate_GIC2, Calculate_GIC2_short,
       Calculate_GIC3, Calculate_GIC3_short,
       Calculate_GIC4, Calculate_GIC4_short,
       Calculate_GIC5, Calculate_GIC5_short,
       Calculate_GIC6, Calculate_GIC6_short,
       Calculate_EBIC, Calculate_EBIC_short,
       GIC_Variable_Selection, GIC_Variable_Selection_Boltzmann,
       Beta_estimate, Y_to_LP, LP_to_Y

end # module
