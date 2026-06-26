# runLattice_source_study.jl
#
# MLMC-DLRA study for the Lattice test case with heterogeneous absorption
# uncertainty (uncertParam "2"): inner blocks carry higher uncertainty
# than outer blocks, source kept at nominal value.
#
#   Case A : outer ±5 %,  inner ±10 %
#   Case B : outer ±10 %, inner ±15 %
#
# sample[1] ~ Uniform(1-p_outer, 1+p_outer)  →  x∈[1,2]∪[5,6]
# sample[2] ~ Uniform(1-p_inner, 1+p_inner)  →  x∈[2,3]∪[4,5] and x∈[3,4]
# sample[3] source  →  Uniform(1-p_src, 1+p_src),  p_src = 10 %
# Results are saved to Results/Lattice/source_study/.

using PyPlot
using JLD2
using Distributions
using LaTeXStrings
using LinearAlgebra

include("../test_problems/radiation_transport/2D_slabgeometry/settings.jl")
include("../test_problems/radiation_transport/2D_slabgeometry/solver.jl")
include("../src/Integrators.jl")
include("../src/uq.jl")
include("../src/plotting.jl")

close("all")
results_dir = joinpath(@__DIR__, "../Results/Lattice/source_study")
mkpath(results_dir)

# =============================================================
# Problem / solver parameters
# =============================================================
problem  = "Lattice"
Nx0      = 2^5 + 2        # coarsest level: 34 nodes
ε        = 1e-2           # target RMSE
N_warmup = 10             # warm-up samples per level per iteration

# Heterogeneous absorption + source uncertainty (uncertParam "2"):
# sample[1] ~ Uniform(1-p_outer, 1+p_outer)  →  outer blocks  (x∈[1,2]∪[5,6] and x∈[3,4],y∈[5,6])
# sample[2] ~ Uniform(1-p_inner, 1+p_inner)  →  inner blocks  (x∈[2,3]∪[4,5])
# sample[3] ~ Uniform(1-p_src,   1+p_src)    →  source strength (x∈[3,4], y∈[3,4])
p_src         = 0.10
hetero_cases  = [(0.05, 0.10, "hetero_outer05_inner10"),  # Case A already done
                 (0.10, 0.15, "hetero_outer10_inner15")]
hetero_labels = [c[3] for c in hetero_cases]

# =============================================================
# Helper: build a fresh level-0 DLRASetup (uncertParam "2")
# =============================================================
function make_level0_src(Nx0, problem)
    s    = settings(Nx0, Nx0, 50, problem)
    Slvr = solver(s)
    Slvr.uncertParam = "2"

    DLRASetup = DLRAIntegratorSetup(Slvr, setupIC, K_step, L_step, S_step,
                                    pre_step, post_step)
    DLRASetup.r  = max(round(Int, min(Slvr.pn.nTotalEntries, Nx0 - 1) / 2), 2)
    DLRASetup.ϑ  = min(s.dx^2, s.dy^2)
    DLRASetup.cη = 5.0
    return DLRASetup
end

# =============================================================
# Helper: 2D log-scale pcolormesh panels for Q_ℓ and ΔQ_ℓ per level
# =============================================================
function plot_level_solutions_src(uqsetup, Dict_levels, label, results_dir)
    LogNorm = PyPlot.matplotlib.colors.LogNorm

    L = maximum([parse(Int, k) for k in keys(Dict_levels)
                 if tryparse(Int, k) !== nothing])

    for (tag, key, title_fn) in [
            ("dQ",  "mean",    ℓ -> ℓ == 0 ?
                                    L"$\mathbb{E}[Q_0]$" :
                                    L"$|\mathbb{E}[\Delta Q_{" * "$ℓ" * L"}]|$")]

        # --- Shared colour scale across all levels ---
        all_fields = []
        for ℓ = 0:L
            s_ℓ    = uqsetup.Problem_levels["$ℓ"]["DLRASetup"].solver.settings
            nx, ny = s_ℓ.NCellsX, s_ℓ.NCellsY
            raw    = Dict_levels["$ℓ"][key]
            field  = isa(raw, Vector{Vector{Float64}}) ? raw[1] : raw
            data2D = reshape(field, nx, ny)
            push!(all_fields, 4π .* abs.(data2D[2:end-1, 2:end-1]))
        end
        all_vals = vcat(vec.(all_fields)...)
        pos      = all_vals[all_vals .> 0]
        vmin     = isempty(pos) ? 1e-12 : max(minimum(pos), 1e-12)
        vmax     = max(maximum(all_vals), vmin * 10.0)
        norm     = LogNorm(vmin=vmin, vmax=vmax)

        for ℓ = 0:L
            s_ℓ    = uqsetup.Problem_levels["$ℓ"]["DLRASetup"].solver.settings
            nx, ny = s_ℓ.NCellsX, s_ℓ.NCellsY
            X = s_ℓ.xMid[2:end-1]' .* ones(nx - 2)
            Y = s_ℓ.yMid[2:end-1]' .* ones(ny - 2)

            fig, ax = subplots(1, 1, figsize=(5.0, 4.5))
            im = ax.pcolormesh(X, Y', all_fields[ℓ+1],
                               norm=norm, shading="auto",cmap="inferno", rasterized=true)
            fig.colorbar(im, ax=ax, shrink=0.9)
            ax.set_xlabel(L"x")
            ax.set_ylabel(L"y")
            fig.suptitle(title_fn(ℓ), fontsize=13, y=1.04)
            fig.tight_layout()
            savefig(joinpath(results_dir, "level_$(tag)_ell$(ℓ)_$(label).pdf"),
                    bbox_inches="tight")
            println("Saved: level_$(tag)_ell$(ℓ)_$(label).pdf")
            close(fig)
        end
    end
end

# =============================================================
# Helper: MLMC mean reconstructed on the finest grid
# =============================================================
function mlmc_mean_src(uqsetup, Dict_levels)
    L          = maximum([parse(Int, k) for k in keys(Dict_levels)
                          if tryparse(Int, k) !== nothing])
    settings_L = uqsetup.Problem_levels["$L"]["DLRASetup"].solver.settings
    nx, ny     = settings_L.NCellsX, settings_L.NCellsY
    x_fine     = settings_L.xMid
    y_fine     = settings_L.yMid

    fineGrid = RectangleGrid(x_fine, y_fine)
    mean_L   = zeros(nx * ny)

    for ℓ = 0:L
        s_ℓ    = uqsetup.Problem_levels["$ℓ"]["DLRASetup"].solver.settings
        x_ℓ    = s_ℓ.xMid
        y_ℓ    = s_ℓ.yMid
        coarse = RectangleGrid(x_ℓ, y_ℓ)
        mean_ℓ = Dict_levels["$ℓ"]["mean"][1]

        y_interp = zeros(nx, ny)
        for i = 1:nx
            for j = 1:ny
                z = fineGrid[i, j]
                y_interp[i, j] = GridInterpolations.interpolate(coarse, mean_ℓ, z)
            end
        end
        mean_L .+= vec(y_interp)
    end
    return reshape(mean_L, nx, ny), x_fine, y_fine
end

# =============================================================
# Run MLMC for each heterogeneous absorption case
# =============================================================
all_Dict_levels = Dict{String, Dict}()
all_uqsetups    = Dict{String, UQSetup}()

for (p_outer, p_inner, label) in hetero_cases
    println("\n" * "="^60)
    println("Outer ±$(round(Int, p_outer*100)) %  |  Inner ±$(round(Int, p_inner*100)) %  |  Source ±$(round(Int, p_src*100)) %  (label = $label)")
    println("="^60)

    # sample[1]=outer absorption, sample[2]=inner absorption, sample[3]=source strength
    pdf = [Distributions.Uniform(1.0 - p_outer, 1.0 + p_outer),
           Distributions.Uniform(1.0 - p_inner, 1.0 + p_inner),
           Distributions.Uniform(1.0 - p_src,   1.0 + p_src)]

    DLRASetup0     = make_level0_src(Nx0, problem)
    Problem_levels = Dict("0" => Dict("DLRASetup" => DLRASetup0,
                                       "rank_diff"  => 0,
                                       "rank_sum"   => 0))

    uqsetup = UQSetup("AugBUG", Problem_levels, pdf, "ScalarFlux",
                      "MLMC_adaptive", ε, "2D", N_warmup)
    uqsetup.Plot    = [true, true]
    uqsetup.Verbose = true
    uqsetup.label = label

    Dict_levels = run(uqsetup)

    all_Dict_levels[label] = Dict_levels
    all_uqsetups[label]    = uqsetup

    plot_MLMCParams(Dict_levels, problem, ε)
    plot_level_solutions_src(uqsetup, Dict_levels, label, results_dir)
end

#jldsave(joinpath(results_dir, "hetero_absorption_results.jld2"); all_Dict_levels)
#println("Results saved to $(joinpath(results_dir, "hetero_absorption_results.jld2"))")

markers = ["o", "s"]
colors  = ["#0072B2", "#D55E00"]

# helper: short label string for titles/legends
case_title(k) = "outer ±$(round(Int, hetero_cases[k][1]*100)) %, inner ±$(round(Int, hetero_cases[k][2]*100)) %"

#= Comparison plots commented out — need both cases to run
# =============================================================
# Comparison Figure 1: Mean scalar flux (1×2 panel)
# =============================================================
fig1, axes1 = subplots(1, 2, figsize=(10, 4.5))

for (k, label) in enumerate(hetero_labels)
    DL  = all_Dict_levels[label]
    uqs = all_uqsetups[label]
    mu, xf, yf = mlmc_mean_src(uqs, DL)

    X = xf[2:end-1]' .* ones(length(xf[2:end-1]))
    Y = yf[2:end-1]' .* ones(length(yf[2:end-1]))

    ax    = axes1[k]
    field = 4π .* mu[2:end-1, 2:end-1]
    pos   = field[field .> 0]
    vmin  = isempty(pos) ? 1e-12 : max(minimum(pos), 1e-12)
    vmax  = max(maximum(field), vmin * 10.0)
    im = ax.pcolormesh(X, Y', field, shading="auto", rasterized=true, cmap="inferno",
                       norm=PyPlot.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig1.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(case_title(k), fontsize=11)
    ax.set_xlabel(L"$x$", fontsize=13); ax.set_ylabel(L"$y$", fontsize=13)
    ax.tick_params(labelsize=11)
end

fig1.tight_layout()
savefig(joinpath(results_dir, "comparison_mean_flux.pdf"), bbox_inches="tight")
println("Saved: comparison_mean_flux.pdf")

# =============================================================
# Comparison Figure 2: Variance per MLMC level
# =============================================================
fig2, ax2 = subplots(figsize=(6, 4.5))

for (k, label) in enumerate(hetero_labels)
    DL  = all_Dict_levels[label]
    L   = maximum([parse(Int, ki) for ki in keys(DL) if tryparse(Int, ki) !== nothing])
    var_ℓ = [DL["$ℓ"]["var"] for ℓ = 0:L]
    ax2.semilogy(0:L, var_ℓ, markers[k] * "--",
                 color=colors[k], linewidth=1.6, markersize=7, label=case_title(k))
end

ax2.set_xlabel(L"\ell"); ax2.set_ylabel(L"\mathrm{Var}[\Delta Q_\ell]")
ax2.xaxis.set_major_locator(PyPlot.matplotlib.ticker.MaxNLocator(integer=true))
ax2.legend(fontsize=12); ax2.grid(linestyle="dotted")
fig2.tight_layout()
savefig(joinpath(results_dir, "comparison_variance.pdf"), bbox_inches="tight")
println("Saved: comparison_variance.pdf")

# =============================================================
# Comparison Figure 3: Mean rank per MLMC level
# =============================================================
fig3, ax3 = subplots(figsize=(6, 4.5))

for (k, label) in enumerate(hetero_labels)
    DL    = all_Dict_levels[label]
    uqs   = all_uqsetups[label]
    L     = maximum([parse(Int, ki) for ki in keys(DL) if tryparse(Int, ki) !== nothing])
    N_ℓ   = [DL["$ℓ"]["N_samples"] for ℓ = 0:L]
    r_sum = [uqs.Problem_levels["$ℓ"]["rank_sum"]  for ℓ = 0:L]
    r_max = [uqs.Problem_levels["$ℓ"]["rank_diff"] for ℓ = 0:L]
    mean_rank = r_sum ./ max.(N_ℓ, 1)

    ax3.semilogy(0:L, mean_rank, markers[k] * "--",
                 color=colors[k], linewidth=1.6, markersize=7,
                 label=case_title(k) * " (mean)")
    ax3.semilogy(0:L, r_max, markers[k] * ":",
                 color=colors[k], linewidth=1.0, markersize=5, alpha=0.6,
                 label=case_title(k) * " (max)")
end

ax3.set_xlabel(L"\ell"); ax3.set_ylabel("Rank  " * L"r")
ax3.xaxis.set_major_locator(PyPlot.matplotlib.ticker.MaxNLocator(integer=true))
ax3.legend(ncol=2, fontsize=12); ax3.grid(linestyle="dotted")
fig3.tight_layout()
savefig(joinpath(results_dir, "comparison_rank.pdf"), bbox_inches="tight")
println("Saved: comparison_rank.pdf")

# =============================================================
# Comparison Figure 4: Number of samples per MLMC level
# =============================================================
fig4, ax4 = subplots(figsize=(6, 4.5))

for (k, label) in enumerate(hetero_labels)
    DL  = all_Dict_levels[label]
    L   = maximum([parse(Int, ki) for ki in keys(DL) if tryparse(Int, ki) !== nothing])
    N_ℓ = [DL["$ℓ"]["N_samples"] for ℓ = 0:L]
    ax4.semilogy(0:L, N_ℓ, markers[k] * "--",
                 color=colors[k], linewidth=1.6, markersize=7, label=case_title(k))
end

ax4.set_xlabel(L"\ell"); ax4.set_ylabel(L"N_\ell")
ax4.xaxis.set_major_locator(PyPlot.matplotlib.ticker.MaxNLocator(integer=true))
ax4.legend(fontsize=12); ax4.grid(linestyle="dotted")
fig4.tight_layout()
savefig(joinpath(results_dir, "comparison_Nsamples.pdf"), bbox_inches="tight")
println("Saved: comparison_Nsamples.pdf")

println("\nAll results saved to Results/Lattice/source_study/")
=#
