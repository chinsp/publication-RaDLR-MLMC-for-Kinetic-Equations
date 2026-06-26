# include("../test_problems/shallow_water/solver_SW.jl") # Load the solver
# include("../test_problems/shallow_water/settings_SW.jl") # Load the settings
include("../src/uq.jl")



using PyPlot
using DelimitedFiles
using BenchmarkTools
using LaTeXStrings
using Distributions
using ProgressBars


rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 30


d1 = Distributions.Uniform(-0.5,0.5); # The distribution of the positional uncertainty
d2 = Distributions.Uniform(0.8,1.2); # The distribution of the amplitude uncertainty

pdf = [d2]; # The distribution of the random variable



# Problem_levels = Dict("0" => Dict("settings" => settings, "solver" => solver));  # You need to define settings and solver before this line

uqsetup = UQSetup(100,"full",Problem_levels, pdf,"ScalarFlux","MLMC_adaptive",1e-2); # Create the UQ setup with the specified parameters
Dict_levels = run(uqsetup);
plot_MLMCParams(Dict_levels);