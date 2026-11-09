# This MUST be the first line
__precompile__(true)

module GICVariableSelection

# Standard libraries (NO need to declare in Project.toml)
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Distributed

# Regular packages
using CSV
using DataFrames
using Distributions
using IJulia
using LogExpFunctions
using Plots
using StatsBase
using StatsPlots
using StatsModels
using GLM
# Include your files
include("GIC_Calculation.jl")
include("GIC_Fast.jl")
include("GIC_Model_Selection.jl")
include("GIC_Model_Selection_Boltzmann.jl")
include("Beta_estimate.jl")
include("LP_to_Y.jl")
include("Y_to_LP.jl")

# Export your public API
export Calculate_AIC, Calculate_AIC_short,
Calculate_AIC_c, Calculate_AIC_c_short,
Calculate_BIC, Calculate_BIC_short,
Calculate_CAIC, Calculate_CAIC_short,
Calculate_CAICF, Calculate_CAICF_short,
Calculate_SIC, Calculate_SIC_short,
Calculate_AttIC, Calculate_AttIC_short,
Calculate_GIC2, Calculate_GIC2_short,
Calculate_GIC3, Calculate_GIC3_short,
Calculate_GIC4, Calculate_GIC4_short,
Calculate_GIC5, Calculate_GIC5_short,
Calculate_GIC6, Calculate_GIC6_short,
Calculate_ICOMP, Calculate_ICOMP_short,
Calculate_ICOMPIFIM, Calculate_ICOMPIFIM_short,
GIC_Variable_Selection, GIC_Variable_Selection_Boltzmann,
Beta_estimate, Y_to_LP, LP_to_Y

end # module


