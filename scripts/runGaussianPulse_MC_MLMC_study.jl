# runGaussianPulse_MC_MLMC_study.jl
#
# Fair comparison of MC and MLMC for the Gaussian Pulse problem against
# the semi-analytic solution.
#
# Fairness: both methods are evaluated at the same spatial resolution.
# MLMC_adaptive with tolerance ε creates L levels (level 0 = coarsest).
# MC_adaptive is then run on the *same finest-level grid* used by MLMC
# so that spatial discretization errors are identical and the comparison
# purely reflects statistical efficiency.
#
# Evaluation metrics
# ------------------
#   1. L2 mean error
#        e_mean = sqrt( (E[ϕ]_est - E[ϕ]_ref)ᵀ (E[ϕ]_est - E[ϕ]_ref) · dx )
#      Measures accuracy of the expected scalar flux vs the semi-analytic
#      gPC reference. Both methods target RMSE ≤ ε, so we verify e_mean ≤ ε.
#
#   2. Relative integrated variance error  (MC only)
#        e_var = |∫ Var_MC[ϕ] dx  −  ∫ Var_ref[ϕ] dx| / ∫ Var_ref[ϕ] dx
#      For MC, Dict_levels["0"]["var"] = ∫ Var_sample[ϕ] dx exactly
#      (Welford accumulation with spatial dot-product collapses space).
#      For MLMC, Dict_levels["ℓ"]["var"] is the variance of the level-ℓ
#      *difference* estimator, not the solution variance, so the comparison
#      is not directly applicable.
#
#   3. Total solver cost
#        cost = Σ_ℓ  N_ℓ · c_ℓ   (wall-clock seconds)
#      MC cost = N · c  where c is cost per sample on the finest grid.
#
# Output figures (saved to Results/GaussianPulse/)
# ------------------
#   TolVsError_MC_MLMC.pdf   — e_mean vs 1/ε  for MC, Full-MLMC, DLR-MLMC  (ε/√2 reference)
#   CostVsError_MC_MLMC.pdf  — e_mean vs cost  (efficiency comparison)
#   VarError_MC.pdf          — e_var vs 1/ε   (MC variance accuracy vs reference)

using PyPlot
using Printf
using LaTeXStrings
using LinearAlgebra
using Distributions
using Interpolations
using JLD2

include("../test_problems/radiation_transport/1D_slabgeometry/settings.jl")
include("../test_problems/radiation_transport/1D_slabgeometry/solver.jl")
include("../test_problems/radiation_transport/1D_slabgeometry/reference.jl")
include("../src/Integrators.jl")
include("../src/uq.jl")
include("../src/plotting.jl")

# -----------------------------------------------------------------------
# Matplotlib style  (journal two-column, ~3.5 in wide)
# -----------------------------------------------------------------------
close("all")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

rcParams["font.family"]           = "serif"
rcParams["font.serif"]            = ["STIXGeneral", "Computer Modern Roman", "Times New Roman"]
rcParams["mathtext.fontset"]      = "stix"

rcParams["font.size"]             = 11.0
rcParams["axes.labelsize"]        = 13.0
rcParams["axes.titlesize"]        = 13.0
rcParams["xtick.labelsize"]       = 11.0
rcParams["ytick.labelsize"]       = 11.0
rcParams["legend.fontsize"]       = 10.5
rcParams["legend.title_fontsize"] = 10.5

rcParams["axes.linewidth"]        = 0.7
rcParams["xtick.major.width"]     = 0.7
rcParams["ytick.major.width"]     = 0.7
rcParams["xtick.minor.width"]     = 0.45
rcParams["ytick.minor.width"]     = 0.45
rcParams["xtick.major.size"]      = 4.0
rcParams["ytick.major.size"]      = 4.0
rcParams["xtick.minor.size"]      = 2.2
rcParams["ytick.minor.size"]      = 2.2
rcParams["xtick.direction"]       = "in"
rcParams["ytick.direction"]       = "in"
rcParams["xtick.top"]             = true
rcParams["ytick.right"]           = true

rcParams["lines.linewidth"]       = 1.5
rcParams["lines.markersize"]      = 5.5

rcParams["legend.framealpha"]     = 0.95
rcParams["legend.edgecolor"]      = "#cccccc"
rcParams["legend.borderpad"]      = 0.4
rcParams["legend.labelspacing"]   = 0.25
rcParams["legend.handlelength"]   = 1.6

rcParams["figure.dpi"]            = 150
rcParams["savefig.dpi"]           = 600
rcParams["savefig.bbox"]          = "tight"
rcParams["savefig.pad_inches"]    = 0.02

# Colorblind-safe palette (Okabe–Ito)
COL_MC        = "#0072B2"   # blue
COL_FULL_MLMC = "#D55E00"   # vermilion
COL_DLR_MLMC  = "#009E73"   # green
COL_REF       = "black"

W1   = 3.5   # single-column width (inches)
W2   = 7.2   # double-column width (inches)
HASP = 0.75  # height-to-width aspect ratio

# -----------------------------------------------------------------------
# Semi-analytic reference solution (gPC, N = 4 Legendre modes)
#   Uncertainty: σ_A = 1 + θ/10, θ ~ Uniform(-1,1), σ_S = 1, t = 1, σ_G = 0.5
# -----------------------------------------------------------------------
s_ref = settings(2^10 + 1, 501)
x_ref = s_ref.x
println("Computing semi-analytic reference (gPC, N=4)…")
mean_ref, var_ref = computeMoments_GaussianPulse_Uniform(1.0, x_ref, 1.0, 0.5, 4)

# IC is exp(-x²/σ²)/√2, so the gPC moments carry an implicit 1/√2 factor
mean_ref_sa        = mean_ref ./ sqrt(2)
var_ref_integrated = sum(var_ref) * s_ref.dx    # ∫ Var_ref[ϕ(x)] dx  (scalar)
norm_ref           = sqrt(mean_ref_sa' * mean_ref_sa * s_ref.dx)

println("  ||E[ϕ]_ref||_L2 = ", round(norm_ref, sigdigits=4))
println("  ∫ Var_ref[ϕ] dx = ", round(var_ref_integrated, sigdigits=4))

mean_ref_interp = linear_interpolation(x_ref, mean_ref_sa, extrapolation_bc = Line())

# -----------------------------------------------------------------------
# Uncertainty and coarsest-level solver (level ℓ = 0)
# MLMC will auto-create finer levels via check_level_setup_DLR
# -----------------------------------------------------------------------
d               = Distributions.Uniform(-1.0, 1.0)
uncertainty_pdf = [d]
problem         = "GaussianPulse"

function make_level0_setup()
    s0           = settings(2^4 + 1, 501)
    Slv          = solver(s0)
    Slv.uncertParam = "0"
    DS           = DLRAIntegratorSetup(Slv, setupIC, K_step, L_step, S_step, pre_step, post_step)
    DS.r         = max(round(Int, min(s0.Nx, s0.nPN) / 2), 2)
    DS.ϑ         = (minimum(Slv.gridWidth))^2
    DS.cη        = 5.0
    return DS
end

function make_level_setup(Nx_in)
    s0           = settings(Nx_in, 501)
    Slv          = solver(s0)
    Slv.uncertParam = "0"
    DS           = DLRAIntegratorSetup(Slv, setupIC, K_step, L_step, S_step, pre_step, post_step)
    DS.r         = max(round(Int, min(s0.Nx, s0.nPN) / 2), 2)
    DS.ϑ         = (minimum(Slv.gridWidth))^2
    DS.cη        = 5.0
    return DS
end

# -----------------------------------------------------------------------
# Helper: index of highest integer key in Dict_levels
# -----------------------------------------------------------------------
function max_level(D)
    L = 0
    for k in keys(D)
        v = tryparse(Int, k)
        v !== nothing && v > L && (L = v)
    end
    return L
end

# -----------------------------------------------------------------------
# Helper: L2 mean error against semi-analytic reference
# -----------------------------------------------------------------------
function l2_mean_error(mean_est, x_est)
    dx_est = x_est[2] - x_est[1]
    if length(x_ref) >= length(x_est)
        ref = mean_ref_interp(x_est)
    else
        ref_fine, _ = computeMoments_GaussianPulse_Uniform(1.0, x_est, 1.0, 0.5, 4)
        ref = ref_fine ./ sqrt(2)
    end
    return sqrt((mean_est - ref)' * (mean_est - ref) * dx_est)
end


# -----------------------------------------------------------------------
# Tolerance sweep  –  predefining storage vectors
# -----------------------------------------------------------------------

# 1/ε grid: one point in [10,100], four points in [100,1000]
ε_list = [1e-1, 1e-2, 5e-3, 2e-3, 1e-3]
nε     = length(ε_list)
e_mean_MC        = zeros(nε)
e_mean_MLMC      = zeros(nε)   # RaDLR-MLMC (AugBUG)
e_mean_MLMC_full = zeros(nε)   # Full-rank MLMC
cost_MC          = zeros(nε)
cost_MLMC        = zeros(nε)
cost_MLMC_full   = zeros(nε)
N_tot_MC         = zeros(Int, nε)
var_MC           = zeros(nε)   # ∫ Var_MC[ϕ] dx
nRepeat          = 5

# -----------------------------------------------------------------------
# Tolerance sweep  –  load pre-computed results from JLD2
# -----------------------------------------------------------------------
# _res             = load("Results/GaussianPulse/GaussianPulse_MC_MLMC_results.jld2")
# ε_list           = _res["ε_list"]
# nε               = length(ε_list)
# e_mean_MC        = _res["e_mean_MC"]
# e_mean_MLMC_full = _res["e_mean_MLMC_full"]
# e_mean_MLMC      = _res["e_mean_MLMC"]
# cost_MC          = _res["cost_MC"]
# cost_MLMC_full   = _res["cost_MLMC_full"]
# cost_MLMC        = _res["cost_MLMC"]
# e_var_rel_MC     = _res["e_var_rel_MC"]

# -----------------------------------------------------------------------
# Main loop: for each ε, repeat nRepeat times.
#   (1) run RaDLR-MLMC_adaptive  → determines finest level grid Nx_L
#   (2) run Full-rank MLMC on the same levels
#   (3) run MC_adaptive on the same Nx_L grid  → fair comparison
# Errors are averaged over repeats; costs use the minimum (best-case timing).
# -----------------------------------------------------------------------
for (i, ε) in enumerate(ε_list)
    println("\n===  ε = $ε  ===")

    costs_MLMC_reps      = zeros(nRepeat)
    costs_MLMC_full_reps = zeros(nRepeat)
    costs_MC_reps        = zeros(nRepeat)
    errors_MLMC_reps     = zeros(nRepeat)
    errors_MLMC_full_reps = zeros(nRepeat)
    errors_MC_reps       = zeros(nRepeat)
    vars_MC_reps         = zeros(nRepeat)
    N_MC_reps            = zeros(Int, nRepeat)

    for rep = 1:nRepeat
        println("  --- repeat $rep / $nRepeat ---")

        # ---- (1) RaDLR-MLMC run ----
        Problem_levels_MLMC = Dict(
            "0"   => Dict("DLRASetup" => make_level0_setup(), "rank_diff" => 0, "rank_sum" => 0),
            "ref" => Dict("x" => x_ref, "mean" => mean_ref_sa,
                          "var" => var_ref_integrated)
        )
        uq_MLMC         = UQSetup("AugBUG", Problem_levels_MLMC, uncertainty_pdf,
                                   "ScalarFlux", "MLMC_adaptive", ε, "1D", 10)
        uq_MLMC.Plot    = [false, false]
        uq_MLMC.Verbose = false

        Dict_MLMC       = run(uq_MLMC)
        mean_MLMC_est   = ComputeMean_DLR(uq_MLMC, Dict_MLMC)
        L_MLMC          = max_level(Dict_MLMC)

        Nx_fine = uq_MLMC.Problem_levels["$L_MLMC"]["DLRASetup"].solver.settings.Nx
        x_fine  = uq_MLMC.Problem_levels["$L_MLMC"]["DLRASetup"].solver.settings.x

        errors_MLMC_reps[rep] = l2_mean_error(mean_MLMC_est, x_fine)
        costs_MLMC_reps[rep]  = sum(Dict_MLMC["$ℓ"]["N_samples"] * Dict_MLMC["$ℓ"]["cost"]
                                    for ℓ = 0:L_MLMC)

        N_str = join(["N_$ℓ=$(Dict_MLMC["$ℓ"]["N_samples"])" for ℓ = 0:L_MLMC], ",  ")
        println("    RaDLR-MLMC: Nx=$Nx_fine,  $N_str")
        println("                e_mean=$(round(errors_MLMC_reps[rep],sigdigits=3)),  cost=$(round(costs_MLMC_reps[rep],sigdigits=3)) s")

        # ---- (2) Full-rank MLMC on the same levels ----
        Problem_levels_MLMC_full = Dict(
            "0"   => Dict("DLRASetup" => make_level0_setup(), "rank_diff" => 0, "rank_sum" => 0),
            "ref" => Dict("x" => x_ref, "mean" => mean_ref_sa,
                          "var" => var_ref_integrated)
        )
        uq_MLMC_full         = UQSetup("Full", Problem_levels_MLMC_full, uncertainty_pdf,
                                        "ScalarFlux", "MLMC_adaptive", ε, "1D", 10)
        uq_MLMC_full.Plot    = [false, false]
        uq_MLMC_full.Verbose = false

        Dict_MLMC_full       = run(uq_MLMC_full)
        mean_MLMC_full_est   = ComputeMean_DLR(uq_MLMC_full, Dict_MLMC_full)
        L_MLMC_full          = max_level(Dict_MLMC_full)

        errors_MLMC_full_reps[rep] = l2_mean_error(mean_MLMC_full_est, x_fine)
        costs_MLMC_full_reps[rep]  = sum(Dict_MLMC_full["$ℓ"]["N_samples"] * Dict_MLMC_full["$ℓ"]["cost"]
                                         for ℓ = 0:L_MLMC_full)

        N_str_full = join(["N_$ℓ=$(Dict_MLMC_full["$ℓ"]["N_samples"])" for ℓ = 0:L_MLMC_full], ",  ")
        println("    Full-MLMC:  $N_str_full")
        println("                e_mean=$(round(errors_MLMC_full_reps[rep],sigdigits=3)),  cost=$(round(costs_MLMC_full_reps[rep],sigdigits=3)) s")

        # ---- (3) MC run on the same finest grid ----
        DS_MC   = make_level_setup(Nx_fine)
        DS_MC.ϑ = uq_MLMC.Problem_levels["$L_MLMC"]["DLRASetup"].ϑ
        Problem_levels_MC = Dict(
            "0"   => Dict("DLRASetup" => DS_MC, "rank_diff" => 0, "rank_sum" => 0),
            "ref" => Dict("x" => x_ref, "mean" => mean_ref_sa,
                          "var" => var_ref_integrated)
        )
        uq_MC         = UQSetup("AugBUG", Problem_levels_MC, uncertainty_pdf,
                                 "ScalarFlux", "MC_adaptive", ε/sqrt(2), "1D", 10)
        uq_MC.Plot    = [false, false]
        uq_MC.Verbose = false

        Dict_MC       = run(uq_MC)
        mean_MC_est   = ComputeMean_DLR(uq_MC, Dict_MC)

        errors_MC_reps[rep] = l2_mean_error(mean_MC_est, x_fine)
        N_MC_reps[rep]      = Dict_MC["0"]["N_samples"]
        costs_MC_reps[rep]  = Dict_MC["0"]["N_samples"] * Dict_MC["0"]["cost"]
        vars_MC_reps[rep]   = Dict_MC["0"]["var"]

        println("    RaDLR-MC:   Nx=$Nx_fine,  N=$(N_MC_reps[rep])")
        println("                e_mean=$(round(errors_MC_reps[rep],sigdigits=3)),  cost=$(round(costs_MC_reps[rep],sigdigits=3)) s")
    end

    # Aggregate: mean error over repeats, minimum cost (best-case timing)
    e_mean_MLMC[i]     = sum(errors_MLMC_reps)      / nRepeat
    e_mean_MLMC_full[i] = sum(errors_MLMC_full_reps) / nRepeat
    e_mean_MC[i]       = sum(errors_MC_reps)         / nRepeat
    cost_MLMC[i]       = minimum(costs_MLMC_reps)
    cost_MLMC_full[i]  = minimum(costs_MLMC_full_reps)
    cost_MC[i]         = minimum(costs_MC_reps)
    N_tot_MC[i]        = N_MC_reps[argmin(costs_MC_reps)]
    var_MC[i]          = vars_MC_reps[argmin(costs_MC_reps)]

    println("  => avg e_mean:  RaDLR-MLMC=$(round(e_mean_MLMC[i],sigdigits=3)),  Full-MLMC=$(round(e_mean_MLMC_full[i],sigdigits=3)),  RaDLR-MC=$(round(e_mean_MC[i],sigdigits=3))")
    println("  => min cost:    RaDLR-MLMC=$(round(cost_MLMC[i],sigdigits=3)),  Full-MLMC=$(round(cost_MLMC_full[i],sigdigits=3)),  RaDLR-MC=$(round(cost_MC[i],sigdigits=3)) s")
end

# -----------------------------------------------------------------------
# Relative integrated variance error (MC vs semi-analytic)
# -----------------------------------------------------------------------
e_var_rel_MC = abs.(var_MC .- var_ref_integrated) ./ var_ref_integrated

# Save tolerance-sweep results so plots can be reproduced without re-running
jldsave("Results/GaussianPulse/GaussianPulse_MC_MLMC_results.jld2";
        ε_list, e_mean_MC, e_mean_MLMC_full, e_mean_MLMC,
        cost_MC, cost_MLMC_full, cost_MLMC, e_var_rel_MC)
println("Results saved to Results/GaussianPulse/GaussianPulse_MC_MLMC_results.jld2")

println("\n--- Summary ---")
println("ε         | e_mean RaDLR-MC | e_mean Full-MLMC | e_mean RaDLR-MLMC | cost RaDLR-MC | cost Full-MLMC | cost RaDLR-MLMC | e_var RaDLR-MC (rel)")
for i = 1:nε
    @printf("%-9.1e | %-15.3e | %-16.3e | %-18.3e | %-13.3e | %-14.3e | %-15.3e | %.3e\n",
            ε_list[i], e_mean_MC[i], e_mean_MLMC_full[i], e_mean_MLMC[i],
            cost_MC[i], cost_MLMC_full[i], cost_MLMC[i], e_var_rel_MC[i])
end

# Shared axis-polish helper
function polish!(ax; minorticks=true)
    minorticks && ax.minorticks_on()
    ax.tick_params(which="both", top=true, right=true, direction="in")
end

# -----------------------------------------------------------------------
# Figure 1: L2 mean error vs 1/ε
# -----------------------------------------------------------------------
mkpath("Results/$problem")
fig1, ax1 = plt.subplots(figsize=(W1, W1 * HASP))

ax1.loglog(1 ./ ε_list, e_mean_MC,
           color=COL_MC,        marker="o", linestyle="-",
           markerfacecolor=COL_MC, markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{RaDLR-MC}")
ax1.loglog(1 ./ ε_list, e_mean_MLMC_full,
           color=COL_FULL_MLMC, marker="^", linestyle="-",
           markerfacecolor=COL_FULL_MLMC, markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{Full-MLMC}")
ax1.loglog(1 ./ ε_list, e_mean_MLMC,
           color=COL_DLR_MLMC,  marker="s", linestyle="--",
           markerfacecolor=COL_DLR_MLMC,  markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{RaDLR-MLMC}")
ax1.loglog(1 ./ ε_list, ε_list ./ sqrt(2),
           color=COL_REF, linestyle=":", linewidth=1.0,
           label=L"\varepsilon/\!\sqrt{2}")

ax1.set_xlabel(L"1/\varepsilon")
ax1.set_ylabel("relative error")
ax1.legend(loc="best", fontsize=12)
polish!(ax1)
fig1.tight_layout(pad=0.3)
fig1.savefig("Results/$problem/TolVsError_MC_MLMC.pdf")
println("\nSaved Results/$problem/TolVsError_MC_MLMC.pdf")

# -----------------------------------------------------------------------
# Figure 2: L2 mean error vs total solver cost  (efficiency)
# -----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(W1, W1 * HASP))

ax2.loglog(cost_MC,
           e_mean_MC,
           color=COL_MC,        marker="o", linestyle="-",
           markerfacecolor=COL_MC,        markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{RaDLR-MC}")
ax2.loglog(cost_MLMC_full,
           e_mean_MLMC_full,
           color=COL_FULL_MLMC, marker="^", linestyle="-",
           markerfacecolor=COL_FULL_MLMC, markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{Full-MLMC}")
ax2.loglog(cost_MLMC,
           e_mean_MLMC,
           color=COL_DLR_MLMC,  marker="s", linestyle="--",
           markerfacecolor=COL_DLR_MLMC,  markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{RaDLR-MLMC}")

ax2.set_xlabel("total cost (runtime in secs)")
ax2.set_ylabel("relative error")
ax2.legend(loc="best", fontsize=12)
polish!(ax2)
fig2.tight_layout(pad=0.3)
fig2.savefig("Results/$problem/CostVsError_MC_MLMC.pdf")
println("Saved Results/$problem/CostVsError_MC_MLMC.pdf")

# -----------------------------------------------------------------------
# Figure 3: MC relative integrated variance error vs 1/ε
# -----------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(W1, W1 * HASP))

ax3.loglog(1 ./ ε_list, e_var_rel_MC,
           color=COL_MC, marker="o", linestyle="-",
           markerfacecolor=COL_MC, markeredgecolor="white", markeredgewidth=0.5,
           label=L"\mathrm{RaDLR-MC}")

ax3.set_xlabel(L"1/\varepsilon")
ax3.set_ylabel(L"\left|\int\widehat{\mathrm{Var}}[\phi]\,\mathrm{d}x - \int\mathrm{Var}_{\mathrm{sa}}[\phi]\,\mathrm{d}x\right|/\int\mathrm{Var}_{\mathrm{sa}}[\phi]\,\mathrm{d}x")
ax3.legend(loc="best", fontsize=12)
polish!(ax3)
fig3.tight_layout(pad=0.3)
fig3.savefig("Results/$problem/VarError_MC.pdf")
println("Saved Results/$problem/VarError_MC.pdf")

println("\nDone.")
