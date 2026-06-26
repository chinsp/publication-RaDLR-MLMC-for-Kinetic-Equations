flush(stdout)
T = Float32;
using LaTeXStrings
include("uq_rt.jl")
include("plotting.jl")

#comment this out if running several times
import Pkg
Pkg.activate("TITUS/.")

using TITUS

using PyPlot
using DelimitedFiles
using BenchmarkTools
using LaTeXStrings
using Distributions
using ProgressBars
using CUDA
using JLD2
using TOML
#identify least used GPU and switch to that
devs = CUDA.devices()
mem_free = Float64[]

for (i,dev) in enumerate(devs)
    CUDA.device!(dev)

    # This forces context initialization
    CUDA.zeros(1)

    # Get free and total memory in bytes
    free, total = CUDA.memory_info()
    push!(mem_free, free)

    println("Device $(i -1): ", 
            " - Free memory: ", round(free / 1_048_576, digits=2), " MB / ",
            round(total / 1_048_576, digits=2), " MB")
end

# Select device with most free memory
best_idx = argmax(mem_free)
best_dev = collect(devs)[best_idx]

CUDA.device!(best_dev)
println("Selected GPU: $(best_idx - 1)")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 30

println("Finished loading packages, starting with problem set up")
flush(stdout)


d1 = Distributions.Normal(0.0,0.3); # The distribution of the positional uncertainty x
d2 = Distributions.Normal(0.0,0.3); # The distribution of the positional uncertainty y
pdf = [d1,d2]; # The distribution of the random variable

file_path = "configFiles/config_CT.toml"
config = TOML.parsefile(file_path)
trace = get(config["computation"], "trace", "false")
disableGPU = get(config["computation"], "disableGPU", "false")
mu_e = get(config["physics"], "eKin", 90)
collided_model = get(config["physics"], "model", "Boltzmann")
file_name = split(file_path, ".")[1]
order = get(config["numerics"], "order", 2)
gridScale = 1.0

close("all")

info = "CUDA"
s = TITUS.Settings(file_path);
if CUDA.functional() && ~disableGPU
    solver1 = TITUS.SolverGPU(s,order);
else
    solver1 = TITUS.SolverCPU(s,order);
end
dose = TITUS.Solve_sample(solver1,T.([0.5,0.0])); #test deterministic solver
Problem_levels = Dict("0" => Dict("settings" => s, "solver" => solver1));

fig = figure()
pcolormesh(reshape(dose,s.NCellsX,s.NCellsY,s.NCellsZ)[:,:,Int(round(s.NCellsZ/2))])
savefig("output/dose_1Sample.png")

#coarse MC
# Nsamples=50
# mean_MC = zeros(size(dose))
# for i = 1:Nsamples
#     alpha_i = [rand(pdf[j]) for j = 1:length(pdf)];
#     dose_MC =  TITUS.Solve_sample(solver1,T.(alpha_i)); 
#     mean_MC .+= dose_MC./Nsamples
# end

# fig = figure()
# pcolormesh(reshape(mean_MC,s.NCellsX,s.NCellsY,s.NCellsZ)[Int(round(s.NCellsX/2)),:,:])
# savefig("dose_coarseMC.png")

println("Start UQ set up")
flush(stdout)
uqsetup = UQSetup(50,"csdAugBUG",Problem_levels, pdf,"FullSolution","MLMC_adaptive",T.(0.1),T.([4.0,250000.0,80.0]));
println("Run UQ solver")
flush(stdout)
Dict_levels = run(uqsetup);
settings = s

L = length(Dict_levels)-2; # Exclude the convergence rates from the number of levels
gridScale_L = gridScale*2^L;
settings_L = TITUS.Settings("configFiles/temp.toml");
idxX = Int(ceil(settings_L.NCellsX/2))
idxZ = Int(ceil(settings_L.NCellsZ/2))
Z = (settings_L.zMid'.*ones(size(settings_L.yMid)))
YZ = (settings_L.yMid'.*ones(size(settings_L.zMid)))'
X = (settings_L.xMid'.*ones(size(settings_L.yMid)))
XY = (settings_L.yMid'.*ones(size(settings_L.xMid)))'
mean_total = load("output/mean_total.jld")["mean_total"]

close("all");
plt.figure(figsize=(settings.d, settings.f));
#plt.pcolormesh(YZ',Z',settings_L.density[idxX,:,:]',vmin = 0.0, vmax =1.85, cmap="gray")
# plt.pcolormesh(YZ',Z',mean_total[idxX,:,:]',alpha=Float64.(mean_total[idxX,:,:].>0.05*maximum(mean_total[idxX,:,:]))',vmin=0,vmax=maximum(mean_total[idxX,:,:]),cmap="jet");
plt.pcolormesh(YZ',Z',mean_total[idxX,:,:]')
plt.grid(linestyle="dotted");
plt.title(string("Mean solution"));
plt.savefig("output/MLMC_ExpectedVal_$(settings.problem)_$(settings.particle).png",bbox_inches="tight");            

plot_MLMCParams(Dict_levels,"smallCT");

close("all");
plt.figure(figsize=(10, 10));
plt.pcolormesh(X',XY',settings_L.density[:,:,idxZ],cmap="gray")
plt.pcolormesh(X',XY',(mean_total[:,:,idxZ].+mean_total[:,:,idxZ+1]),alpha=Float64.(0.5.*mean_total[:,:,idxZ].>0.05*maximum(mean_total[:,:,idxZ])),vmin=0,vmax=maximum(mean_total[:,:,idxZ]),cmap="jet");
plt.grid(linestyle="dotted");
plt.title(string("Mean solution"));
plt.savefig("output/MLMC_ExpectedVal_$(settings.problem)_$(settings.particle)_XY.png",bbox_inches="tight");           


L = length(Dict_levels) - 2
L_plot = 0
ε = uqsetup.ε
problem = settings.problem
path = "output/$problem/$ε"
CheckCreateFolder(path)

# ── Build fine-level settings ────────────────────────────────────────────────
gridScale = 1.0
gridScale_L = gridScale * 2^L_plot
Nx_fine = (settings.Nx - 3) * 2^L_plot + 3
Ny_fine = (settings.Ny - 3) * 2^L_plot + 3
Nz_fine = (settings.Nz - 3) * 2^L_plot + 3
write_config(
    particle=settings.particle,
    problem=settings.problem,
    model=settings.model,
    OmegaMin=0,
    eKin=settings.mu_e,
    nx=Nx_fine,
    ny=Ny_fine,
    nz=Nz_fine,
    nMoments=settings.nPN,
    order=2,
    rank=settings.r,
    maxRank=settings.rMax,
    solverName="",
    tracerFileName="",
    cfl=settings.cfl,
    tolerance=settings.epsAdapt,
    trace=true,
    disableGPU=false)
settings_fine = TITUS.Settings("configFiles/temp.toml")

x_fine = range(settings_fine.xMid[1], settings_fine.xMid[end], length=settings_fine.NCellsX)
y_fine = range(settings_fine.yMid[1], settings_fine.yMid[end], length=settings_fine.NCellsY)
z_fine = range(settings_fine.zMid[1], settings_fine.zMid[end], length=settings_fine.NCellsZ)

# ── Accumulate mean across levels onto fine grid ─────────────────────────────
mean_accumulated = zeros(Float32, settings_fine.NCellsX, settings_fine.NCellsY, settings_fine.NCellsZ)

for l = 0:L_plot
    Nx_l = (settings.Nx - 3) * 2^l + 3
    Ny_l = (settings.Ny - 3) * 2^l + 3
    Nz_l = (settings.Nz - 3) * 2^l + 3
    write_config(
        particle=settings.particle,
        problem=settings.problem,
        model=settings.model,
        OmegaMin=0,
        eKin=settings.mu_e,
        nx=Nx_l,
        ny=Ny_l,
        nz=Nz_l,
        nMoments=settings.nPN,
        order=2,
        rank=settings.r,
        maxRank=settings.rMax,
        solverName="",
        tracerFileName="",
        cfl=settings.cfl,
        tolerance=settings.epsAdapt,
        trace=true,
        disableGPU=false)
    settings_l = TITUS.Settings("configFiles/temp.toml")
    mean_total = load("mean_total_$l.jld")["mean_total"]
    println("Level $l  ‖mean‖ = ", norm(mean_total))

    x_l = range(settings_l.xMid[1], settings_l.xMid[end], length=settings_l.NCellsX)
    y_l = range(settings_l.yMid[1], settings_l.yMid[end], length=settings_l.NCellsY)
    z_l = range(settings_l.zMid[1], settings_l.zMid[end], length=settings_l.NCellsZ)

    itp  = interpolate(mean_total, BSpline(Cubic(Interpolations.Line(OnGrid()))))
    sitp = Interpolations.scale(itp, x_l, y_l, z_l)
    etp  = extrapolate(sitp, 0.0)
    mean_accumulated .+= Float32.(etp(x_fine, y_fine, z_fine))

    # ── Per-level difference panels ──────────────────────────────────────────
    _save_3D_panels(mean_total, settings_l.density, settings_l,
                    path, "MLMC_DeltaQ_$(problem)_$(settings.particle)_level_$(l)",is_difference=true)

end
mean_accumulated_L = mean_accumulated ./sum(mean_accumulated) *settings_fine.dx * settings_fine.dy * settings_fine.dz * sum(settings_fine.mu_e)
_save_3D_panels(mean_accumulated, settings_fine.density, settings_fine,
                path, "MLMC_ExpectedVal_$(problem)_$(settings.particle)_accumulated_L$(L_plot)")


#if needed to load Dict_levels from file:
# @load "output/Dict_levels.jld" dict_levels
# function reconstruct(x)
#     if x isa JLD2.ReconstructedMutable
#         # convert reconstructed dict to a real Dict, recursively
#         return Dict(
#             reconstruct(x.keys[i]) => reconstruct(x.values[i])
#             for i in eachindex(x.keys)
#         )
#     elseif x isa AbstractArray
#         # handle arrays that may contain reconstructed objects
#         return map(reconstruct, x)
#     else
#         # numbers, strings, etc.
#         return x
#     end
# end
# Dict_levels = reconstruct(dict_levels);


