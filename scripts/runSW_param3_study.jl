# runSW_param3_study.jl
#
# MLMC study for the shallow water shock test case with combined uncertainty
# in all three initial-condition parameters simultaneously.
#
# Initial condition:
#   h(t=0, x; ω) = ω₁ + ω₂·(tanh(50x) − tanh(50·(x − 0.2·ω₃)))
# where the three uncertain parameters have nominal values:
#   ω₁ = 0.3   (background height)
#   ω₂ = 0.35  (jump amplitude)
#   ω₃ = 1.0   (shock position scale)
# (uncertParam = "3" in the solver)
#
# Three uncertainty levels (±5 %, ±10 %, ±15 %, ±20 %, ±30 %) and two RMSE
# tolerances (1e-3, 1e-4) are studied with 20 warm-up samples.
#
# QoI: height h ("ScalarFlux"), momentum hu ("Momentum").
# Both are estimated in a single MLMC run (solver called once per sample).

using PyPlot
using Distributions
using LaTeXStrings
using JLD2
using Interpolations

include("../test_problems/shallow_water/settings.jl")
include("../test_problems/shallow_water/solver.jl")
include("../src/uq.jl")
include("../src/plotting.jl")

close("all")

# ─────────────────────────────────────────────────────────────────────────────
# Study parameters
# ─────────────────────────────────────────────────────────────────────────────
problem            = "shock"
Nx0                = 102           # coarse grid: NCells = 101, dx ≈ 0.02
nominal_omega      = [0.3, 0.35, 1.0]   # [background, amplitude, position scale]
uncertainty_pcts   = [0.05, 0.10, 0.15, 0.20, 0.30]
uncertainty_labels = ["5%", "10%", "15%", "20%", "30%"]
tolerances         = [1e-2,5e-3] #1e-1, 
N_warmup           = 20

results_dir = joinpath(@__DIR__, "../Results/ShallowWater/param3_study")
mkpath(results_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: assemble MLMC mean for one FoI (by index) on the finest grid
# ─────────────────────────────────────────────────────────────────────────────
function assemble_mean(uqsetup, Dict_levels, foi_idx::Int)
    L = maximum(parse(Int, k) for k in keys(Dict_levels) if tryparse(Int, k) !== nothing)
    x_fine = uqsetup.Problem_levels["$L"]["solver"].settings.xMid
    mean_q = zeros(length(x_fine))
    for ℓ = 0:L
        x_ℓ = uqsetup.Problem_levels["$ℓ"]["solver"].settings.xMid
        mean_q .+= linear_interpolation(x_ℓ, Dict_levels["$ℓ"]["mean"][foi_idx],
                                         extrapolation_bc=Line()).(x_fine)
    end
    return mean_q, x_fine
end

# ─────────────────────────────────────────────────────────────────────────────
# Run study
# ─────────────────────────────────────────────────────────────────────────────
all_results   = Dict{Tuple{Float64,Float64}, NamedTuple}()
all_uqsetups  = Dict{Tuple{Float64,Float64}, UQSetup}()
all_DL        = Dict{Tuple{Float64,Float64}, Dict}()

for ε in tolerances
    println("\n" * "="^60)
    println("Tolerance  ε = $ε")
    println("="^60)

    for (p, lbl) in zip(uncertainty_pcts, uncertainty_labels)
        println("\n  Uncertainty ±$lbl")

        # One independent uniform distribution per uncertain parameter
        pdf = [Distributions.Uniform(ω * (1.0 - p), ω * (1.0 + p))
               for ω in nominal_omega]

        s      = settings(Nx0; problem = problem)
        s.ϑ    = s.dx^2
        s.rMax = floor(Int, (s.N - 2) / 2)
        Slvr   = solver(s)
        Slvr.uncertParam = "3"
        PL = Dict("0" => Dict("solver" => Slvr, "rank_diff" => 0))

        uqsetup         = UQSetup("AugBUG", PL, pdf,
                                  ["ScalarFlux", "Momentum"],
                                  "MLMC_adaptive", ε, "1D", N_warmup)
        uqsetup.Verbose = true

        Dict_levels = run(uqsetup)

        L       = maximum(parse(Int, k) for k in keys(Dict_levels)
                          if tryparse(Int, k) !== nothing)
        N_total = sum(Dict_levels["$ℓ"]["N_samples"] for ℓ = 0:L)

        mean_h,  x_fine = assemble_mean(uqsetup, Dict_levels, 1)
        mean_hu, _      = assemble_mean(uqsetup, Dict_levels, 2)

        plot_MLMCParams(Dict_levels, problem, ε)

        println("  ±$lbl, ε=$ε  →  L=$L,  total solver calls = $N_total")
        all_results[(ε, p)]  = (h=mean_h, hu=mean_hu, x=x_fine, N=N_total, L=L)
        all_uqsetups[(ε, p)] = uqsetup
        all_DL[(ε, p)]       = Dict_levels
    end
end

jldsave(joinpath(results_dir, "SW_param3_study_results.jld2");
        all_results      = all_results,
        uncertainty_pcts = uncertainty_pcts,
        tolerances       = tolerances)
println("\nRaw results saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Plotting — one figure per tolerance, two subplots (h, hu)
# ─────────────────────────────────────────────────────────────────────────────
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
styles = ["-", "--", ":", "-.", (0,(3,1,1,1))]

qoi_keys  = [:h,  :hu]
qoi_ylbls = [L"h", L"hu"]

for ε in tolerances
    fig, axes = subplots(1, 2, figsize=(11, 4.5))

    for (qi, (qkey, qylbl)) in enumerate(zip(qoi_keys, qoi_ylbls))
        ax = axes[qi]
        for (k, (p, lbl)) in enumerate(zip(uncertainty_pcts, uncertainty_labels))
            res = all_results[(ε, p)]
            ax.plot(res.x, getfield(res, qkey),
                    color=colors[k], linestyle=styles[k],
                    linewidth=2.0, label="±$lbl  (N=$(res.N))")
        end
        # ax.set_xlabel(L"x", fontsize=12)
        # ax.set_ylabel(qylbl, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(linestyle=":")
    end

    tol_str = replace(string(ε), "." => "p")
    fig.tight_layout()

    fname = joinpath(results_dir, "SW_param3_tol$(tol_str).pdf")
    savefig(fname, bbox_inches="tight")
    println("Saved: $fname")
    close(fig)
end

println("\nDone.  All figures saved to $results_dir/")
