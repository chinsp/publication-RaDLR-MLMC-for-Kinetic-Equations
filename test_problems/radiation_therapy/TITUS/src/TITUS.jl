module TITUS

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using TimerOutputs
using JLD2
using Glob
using CUDA
using TOML
using LinearAlgebra
using SparseArrays
using Base: Float64
using Dates
using DICOM
using Libdl
using SparseArrays
using Images, FileIO, ImageTransformations

T = Float32;

# Global state for function parameters
const dqage_state = Dict{Symbol, Any}()
const to = TimerOutput()

include("utils.jl")
include("settings.jl")
include("CSD.jl")
include("quadratures/Quadrature.jl")
include("PNSystem.jl")
include("SolverGPU.jl")
include("SolverCPU.jl")
include("run_simulation.jl")
include("cuda_setup.jl")

export runAndPlot, setup_best_gpu, runAndPlotOctree
export Solve

end