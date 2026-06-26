# runLinesource_tol_study.jl
#
# Full tolerance study for the Linesource test case:
#   • Spatial refinement:  fixed nPN=39,  grids 16×16 → 256×256
#   • Angular refinement:  fixed 256×256, N_PN = 8 → 128
# Results are cached to CSV; solver runs only when CSVs are absent.

using PyPlot
using DelimitedFiles
using LaTeXStrings
using LinearAlgebra
using Printf
using JLD2

BLAS.set_num_threads(1)
const print_lock = ReentrantLock()

include("../test_problems/radiation_transport/2D_slabgeometry/settings.jl")
include("../test_problems/radiation_transport/2D_slabgeometry/solver.jl")
include("../src/Integrators.jl")

# =============================================================
# Publication-quality matplotlib settings
# =============================================================
close("all")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

rcParams["font.family"]           = "serif"
rcParams["font.serif"]            = ["STIXGeneral", "Computer Modern Roman", "Times New Roman"]
rcParams["mathtext.fontset"]      = "stix"
rcParams["font.size"]             = 11.0
rcParams["axes.labelsize"]        = 12.0
rcParams["axes.titlesize"]        = 12.0
rcParams["xtick.labelsize"]       = 10.0
rcParams["ytick.labelsize"]       = 10.0
rcParams["legend.fontsize"]       = 10.0
rcParams["legend.title_fontsize"] = 10.0
rcParams["legend.framealpha"]     = 0.95
rcParams["legend.edgecolor"]      = "#cccccc"
rcParams["legend.borderpad"]      = 0.5
rcParams["legend.labelspacing"]   = 0.3
rcParams["legend.handlelength"]   = 1.8
rcParams["axes.linewidth"]        = 0.75
rcParams["xtick.major.width"]     = 0.75
rcParams["ytick.major.width"]     = 0.75
rcParams["xtick.minor.width"]     = 0.5
rcParams["ytick.minor.width"]     = 0.5
rcParams["xtick.major.size"]      = 4.5
rcParams["ytick.major.size"]      = 4.5
rcParams["xtick.minor.size"]      = 2.5
rcParams["ytick.minor.size"]      = 2.5
rcParams["xtick.direction"]       = "in"
rcParams["ytick.direction"]       = "in"
rcParams["xtick.top"]             = true
rcParams["ytick.right"]           = true
rcParams["lines.linewidth"]       = 1.6
rcParams["lines.markersize"]      = 6.5
rcParams["figure.dpi"]            = 150
rcParams["savefig.dpi"]           = 300
rcParams["savefig.bbox"]          = "tight"
rcParams["savefig.pad_inches"]    = 0.05

# Okabe-Ito palette (colorblind-safe)
GRID_COLORS  = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
GRID_MARKERS = ["o", "s", "^", "D", "v"]

# =============================================================
# Problem parameters
# =============================================================
problem     = "Linesource"
sample      = [1.0, 0.0]
uncertParam = "1"

Nx_list  = [2^k + 1 for k in 4:8]   # [17, 33, 65, 129, 257]
ϑ_list   = 10 .^ range(-0.3, -3.0, length=8)
n_grids  = length(Nx_list)
n_tols   = length(ϑ_list)

nPN_list = [7, 15, 31, 63, 81, 127]
Nx_fixed = 2^8 + 1      # 256×256 cells
n_nPN    = length(nPN_list)

println("Grid sizes (Nx):    ", Nx_list)
println("Tolerances (ϑ):     ", round.(ϑ_list, sigdigits=2))
println("Angular orders (N): ", nPN_list)

results_dir = joinpath(@__DIR__, "../Results")
mkpath(results_dir)

# =============================================================
# Helpers
# =============================================================
function polish_ax(ax; minorticks=true)
    if minorticks; ax.minorticks_on(); end
    ax.tick_params(which="both", top=true, right=true)
end

function add_slope1!(ax, x_vals; color="gray", lw=0.9, fs=9)
    ax.relim(); ax.autoscale_view()
    xs = sort(x_vals)
    x0 = xs[2]
    x1 = x0 * 10^0.5                  # half-decade span
    yl = ax.get_ylim()
    y0 = 10^(log10(yl[1]) + 0.08 * (log10(yl[2]) - log10(yl[1])))
    y1 = y0 * (x1 / x0)               # slope 1: Δlog y = Δlog x
    # triangle: bottom edge, right edge, hypotenuse
    ax.loglog([x0, x1], [y0, y0], "-", color=color, linewidth=lw, zorder=5)
    ax.loglog([x1, x1], [y0, y1], "-", color=color, linewidth=lw, zorder=5)
    ax.loglog([x0, x1], [y0, y1], "-", color=color, linewidth=lw, zorder=5)
    # "1" labels on each leg
    xm = exp10(0.5 * (log10(x0) + log10(x1)))
    ym = exp10(0.5 * (log10(y0) + log10(y1)))
    ax.text(xm,       y0 * 0.60, L"$1$", ha="center", va="top",    fontsize=fs, color=color)
    ax.text(x1 * 1.1, ym,        L"$1$", ha="left",   va="center", fontsize=fs, color=color)
end

# Restrict fine-grid scalar flux to a coarser nested grid via block averaging.
# Storage convention: vectorIndex(N,i,j) = (i-1)*N + j,
# so reshape(phi, Nf, Nf)[j, i] is the value at (xMid[i], yMid[j]).
function restrict_phi(phi_fine::AbstractVector, Nc::Int)
    Nf     = round(Int, sqrt(length(phi_fine)))
    stride = div(Nf, Nc)
    M      = reshape(phi_fine, Nf, Nf)
    out    = zeros(Nc, Nc)
    for I in 1:Nc, J in 1:Nc
        out[J, I] = sum(M[(J-1)*stride+1:J*stride, (I-1)*stride+1:I*stride]) / stride^2
    end
    return vec(out)   # column-major vec: out[J,I] -> J+(I-1)*Nc = vectorIndex(Nc,I,J) ✓
end

# =============================================================
# Compute both studies (spatial + angular) in a single block
# if any CSV is missing.
# =============================================================
spat_csv = joinpath(results_dir, "linesource_tol_errors.csv")
ang_csv  = joinpath(results_dir, "linesource_tol_errors_angular.csv")

need_spatial = !isfile(spat_csv)
need_angular = !isfile(ang_csv)

if need_spatial || need_angular

    # ----------------------------------------------------------
    # Single reference: 256×256 cells, nPN = 127
    # ----------------------------------------------------------
    println("\nReference: $(Nx_fixed-1)×$(Nx_fixed-1) cells, nPN=127 ...")
    s_ref                = settings(Nx_fixed, Nx_fixed, 1, problem)
    s_ref.nPN            = 127
    Slvr_ref             = solver(s_ref)
    Slvr_ref.uncertParam = uncertParam
    Slvr_ref.sample      = sample
    setup_ref            = DLRAIntegratorSetup(Slvr_ref, setupIC, K_step, L_step,
                                               S_step, pre_step, post_step)
    g_ref, _             = FullProblem(setup_ref, sample)
    phi_ref              = g_ref[:, 1]
    dx_ref               = s_ref.dx
    dy_ref               = s_ref.dy
    println("  done.")

    if need_spatial
        errors_spat    = zeros(n_grids, n_tols)
        ranks_spat     = zeros(Int, n_grids, n_tols)
        max_ranks_spat = zeros(Int, n_grids, n_tols)
        spat_done      = falses(n_grids)
        for i in eachindex(Nx_list)
            ckpt_i = joinpath(results_dir, "ckpt_spat_row$(i).jld2")
            if isfile(ckpt_i)
                d = load(ckpt_i)
                errors_spat[i, :]    = d["errors_row"]
                ranks_spat[i, :]     = d["ranks_row"]
                max_ranks_spat[i, :] = d["max_ranks_row"]
                spat_done[i]         = true
                println("  [checkpoint] Loaded spatial row $i (Nx=$(Nx_list[i]))")
            end
        end
    end

    if need_angular
        errors_ang_run    = zeros(n_nPN, n_tols)
        ranks_ang_run     = zeros(Int, n_nPN, n_tols)
        max_ranks_ang_run = zeros(Int, n_nPN, n_tols)
        ang_done          = falses(n_nPN)
        for i in eachindex(nPN_list)
            ckpt_i = joinpath(results_dir, "ckpt_ang_row$(i).jld2")
            if isfile(ckpt_i)
                d = load(ckpt_i)
                errors_ang_run[i, :]    = d["errors_row"]
                ranks_ang_run[i, :]     = d["ranks_row"]
                max_ranks_ang_run[i, :] = d["max_ranks_row"]
                ang_done[i]             = true
                println("  [checkpoint] Loaded angular row $i (N_PN=$(nPN_list[i]))")
            end
        end
    end

    # ----------------------------------------------------------
    # Spatial study: vary Nx, fixed nPN=127
    # ----------------------------------------------------------
    if need_spatial
        println("\n--- Spatial refinement study ---")
        Threads.@threads for i in eachindex(Nx_list)
            spat_done[i] && continue
            Nx = Nx_list[i]
            NCells_lr = Nx - 1
            lock(print_lock) do
                println("  Nx=$Nx  ($(NCells_lr)×$(NCells_lr),  $i/$n_grids)  [thread $(Threads.threadid())]")
            end
            s_lr            = settings(Nx, Nx, 1, problem)
            s_lr.nPN        = 127
            Slvr_lr         = solver(s_lr)
            Slvr_lr.uncertParam = uncertParam
            Slvr_lr.sample      = sample
            r0 = max(round(Int, min(s_lr.NCellsX * s_lr.NCellsY,
                                    Slvr_lr.pn.nTotalEntries) / 2), 2)
            phi_ref_c = (NCells_lr < 256) ? restrict_phi(phi_ref, NCells_lr) : phi_ref

            for (j, ϑ) in enumerate(ϑ_list)
                setup_lr    = DLRAIntegratorSetup(Slvr_lr, setupIC, K_step, L_step,
                                                   S_step, pre_step, post_step)
                setup_lr.r  = r0
                setup_lr.ϑ  = ϑ
                setup_lr.cη = 5.0
                g_lr, rVec = augBUG(setup_lr, sample)
                phi_lr     = g_lr[:, 1]
                errors_spat[i, j]    = sqrt(s_lr.dx * s_lr.dy * sum((phi_lr .- phi_ref_c).^2))
                ranks_spat[i, j]     = Int(rVec[2, end])
                max_ranks_spat[i, j] = Int(maximum(rVec[2, :]))
            end
            jldsave(joinpath(results_dir, "ckpt_spat_row$(i).jld2");
                    errors_row    = errors_spat[i, :],
                    ranks_row     = ranks_spat[i, :],
                    max_ranks_row = max_ranks_spat[i, :])
            lock(print_lock) do
                println("    Nx=$Nx errors: ", round.(errors_spat[i, :], sigdigits=2))
            end
        end
    end

    # ----------------------------------------------------------
    # Angular study: vary N_PN, fixed 256×256 spatial grid
    # ----------------------------------------------------------
    if need_angular
        println("\n--- Angular refinement study ---")
        Threads.@threads for i in eachindex(nPN_list)
            ang_done[i] && continue
            nPN = nPN_list[i]
            lock(print_lock) do
                println("  N_PN=$nPN  ($i/$n_nPN)  [thread $(Threads.threadid())]")
            end
            s_lr     = settings(Nx_fixed, Nx_fixed, 1, problem)
            s_lr.nPN = nPN
            Slvr_lr  = solver(s_lr)
            Slvr_lr.uncertParam = uncertParam
            Slvr_lr.sample      = sample
            r0 = max(round(Int, min(s_lr.NCellsX * s_lr.NCellsY,
                                    Slvr_lr.pn.nTotalEntries) / 2), 2)

            for (j, ϑ) in enumerate(ϑ_list)
                setup_lr    = DLRAIntegratorSetup(Slvr_lr, setupIC, K_step, L_step,
                                                   S_step, pre_step, post_step)
                setup_lr.r  = r0
                setup_lr.ϑ  = ϑ
                setup_lr.cη = 5.0
                g_lr, rVec = augBUG(setup_lr, sample)
                phi_lr     = g_lr[:, 1]
                errors_ang_run[i, j]    = sqrt(dx_ref * dy_ref * sum((phi_lr .- phi_ref).^2))
                ranks_ang_run[i, j]     = Int(rVec[2, end])
                max_ranks_ang_run[i, j] = Int(maximum(rVec[2, :]))
            end
            jldsave(joinpath(results_dir, "ckpt_ang_row$(i).jld2");
                    errors_row    = errors_ang_run[i, :],
                    ranks_row     = ranks_ang_run[i, :],
                    max_ranks_row = max_ranks_ang_run[i, :])
            lock(print_lock) do
                println("    N_PN=$nPN errors: ", round.(errors_ang_run[i, :], sigdigits=2))
            end
        end
    end

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    if need_spatial
        writedlm(joinpath(results_dir, "linesource_tol_errors.csv"),    errors_spat,    ',')
        writedlm(joinpath(results_dir, "linesource_tol_ranks.csv"),     ranks_spat,     ',')
        writedlm(joinpath(results_dir, "linesource_tol_max_ranks.csv"), max_ranks_spat, ',')
        jldsave(joinpath(results_dir, "linesource_tol_spatial_results.jld2");
                Nx_list, ϑ_list, errors_spat, ranks_spat, max_ranks_spat)
        for i in eachindex(Nx_list)
            rm(joinpath(results_dir, "ckpt_spat_row$(i).jld2"); force=true)
        end
        println("\nSpatial data saved.")
    end
    if need_angular
        writedlm(joinpath(results_dir, "linesource_tol_errors_angular.csv"),    errors_ang_run,    ',')
        writedlm(joinpath(results_dir, "linesource_tol_ranks_angular.csv"),     ranks_ang_run,     ',')
        writedlm(joinpath(results_dir, "linesource_tol_max_ranks_angular.csv"), max_ranks_ang_run, ',')
        jldsave(joinpath(results_dir, "linesource_tol_angular_results.jld2");
                nPN_list, ϑ_list,
                errors_ang    = errors_ang_run,
                ranks_ang     = ranks_ang_run,
                max_ranks_ang = max_ranks_ang_run)
        for i in eachindex(nPN_list)
            rm(joinpath(results_dir, "ckpt_ang_row$(i).jld2"); force=true)
        end
        println("Angular data saved.")
    end
end

# =============================================================
# Load results
# =============================================================
errors    = readdlm(joinpath(results_dir, "linesource_tol_errors.csv"),    ',')
ranks     = Int.(readdlm(joinpath(results_dir, "linesource_tol_ranks.csv"),    ','))
max_ranks = Int.(readdlm(joinpath(results_dir, "linesource_tol_max_ranks.csv"), ','))
println("Spatial data loaded.")

errors_ang    = readdlm(joinpath(results_dir, "linesource_tol_errors_angular.csv"),    ',')
ranks_ang     = Int.(readdlm(joinpath(results_dir, "linesource_tol_ranks_angular.csv"),    ','))
max_ranks_ang = Int.(readdlm(joinpath(results_dir, "linesource_tol_max_ranks_angular.csv"), ','))
println("Angular data loaded.")

# Spatial subset: skip 16×16 grid (Nx=17, index 1)
plot_idx      = 2:n_grids
plot_Nx       = Nx_list[plot_idx]
plot_errors   = errors[plot_idx, :]
plot_maxranks = max_ranks[plot_idx, :]
plot_colors   = GRID_COLORS[plot_idx]
plot_markers  = GRID_MARKERS[plot_idx]

ang_colors  = GRID_COLORS[1:n_nPN]
ang_markers = GRID_MARKERS[1:n_nPN]

# =============================================================
# Combined 4-panel figure
# =============================================================
fig, axes = subplots(2, 2, figsize=(11.0, 8.0))
ax_tl = axes[1, 1]   # spatial  tol vs L2 error
ax_tr = axes[1, 2]   # spatial  tol vs max rank
ax_bl = axes[2, 1]   # angular  tol vs L2 error
ax_br = axes[2, 2]   # angular  tol vs max rank

for (i, Nx) in enumerate(plot_Nx)
    NCells = Nx - 1
    lbl    = L"$N_x \times N_y = \,$" * "$NCells" * L"$\times$" * "$NCells"
    kw     = Dict(:color => plot_colors[i], :linewidth => 1.6, :markersize => 6.5,
                  :markerfacecolor => plot_colors[i], :markeredgecolor => "white",
                  :markeredgewidth => 0.6, :label => lbl)
    ax_tl.loglog(  ϑ_list, plot_errors[i, :],   plot_markers[i] * "--"; kw...)
    ax_tr.semilogx(ϑ_list, plot_maxranks[i, :], plot_markers[i] * "--"; kw...)
end
ax_tl.set_xlabel(L"\vartheta");  ax_tl.set_ylabel(L"$\|\phi_r - \phi_{\mathrm{full}}\|_{L^2}$")
ax_tr.set_xlabel(L"\vartheta");  ax_tr.set_ylabel(L"r_{\max}")
for ax in (ax_tl, ax_tr)
    ax.legend(loc = (ax === ax_tl) ? "lower right" : "upper right",
              title=L"$N_x \times N_y$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
    polish_ax(ax)
end

for (i, nPN) in enumerate(nPN_list)
    lbl = L"$N_{PN} = $" * "$nPN"
    kw  = Dict(:color => ang_colors[i], :linewidth => 1.6, :markersize => 6.5,
               :markerfacecolor => ang_colors[i], :markeredgecolor => "white",
               :markeredgewidth => 0.6, :label => lbl)
    ax_bl.loglog(  ϑ_list, errors_ang[i, :],    ang_markers[i] * "--"; kw...)
    ax_br.semilogx(ϑ_list, max_ranks_ang[i, :], ang_markers[i] * "--"; kw...)
end
ax_bl.set_xlabel(L"\vartheta");  ax_bl.set_ylabel(L"$\|\phi_r - \phi_{\mathrm{full}}\|_{L^2}$")
ax_br.set_xlabel(L"\vartheta");  ax_br.set_ylabel(L"r_{\max}")
for ax in (ax_bl, ax_br)
    ax.legend(loc = (ax === ax_bl) ? "lower right" : "upper right",
              title=L"$N_{PN}$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
    polish_ax(ax)
end

add_slope1!(ax_tl, ϑ_list)
add_slope1!(ax_bl, ϑ_list)
fig.tight_layout()
savefig(joinpath(results_dir, "linesource_tol_combined.pdf"))
println("Saved: Results/linesource_tol_combined.pdf")

# =============================================================
# Individual figures — spatial
# =============================================================
fig1, ax1 = subplots(1, 1, figsize=(5.5, 4.2))
for (i, Nx) in enumerate(plot_Nx)
    NCells = Nx - 1
    ax1.loglog(ϑ_list, plot_errors[i, :], plot_markers[i] * "--",
               color=plot_colors[i], linewidth=1.6, markersize=6.5,
               markerfacecolor=plot_colors[i], markeredgecolor="white", markeredgewidth=0.6,
               label=L"$N_x \times N_y = \,$" * "$NCells" * L"$\times$" * "$NCells")
end
ax1.set_xlabel(L"\vartheta");  ax1.set_ylabel(L"$\|\phi_r - \phi_{\mathrm{full}}\|_{L^2}$")
ax1.legend(loc="lower right", title=L"$N_x \times N_y$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
add_slope1!(ax1, ϑ_list)
polish_ax(ax1);  fig1.tight_layout()
savefig(joinpath(results_dir, "linesource_tol_vs_error.pdf"))
println("Saved: Results/linesource_tol_vs_error.pdf")

fig2b, ax2b = subplots(1, 1, figsize=(5.5, 4.2))
for (i, Nx) in enumerate(plot_Nx)
    NCells = Nx - 1
    ax2b.semilogx(ϑ_list, plot_maxranks[i, :], plot_markers[i] * "--",
                  color=plot_colors[i], linewidth=1.6, markersize=6.5,
                  markerfacecolor=plot_colors[i], markeredgecolor="white", markeredgewidth=0.6,
                  label=L"$N_x \times N_y = \,$" * "$NCells" * L"$\times$" * "$NCells")
end
ax2b.set_xlabel(L"\vartheta");  ax2b.set_ylabel(L"r_{\max}")
ax2b.legend(loc="upper right", title=L"$N_x \times N_y$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
polish_ax(ax2b);  fig2b.tight_layout()
savefig(joinpath(results_dir, "linesource_tol_vs_maxrank.pdf"))
println("Saved: Results/linesource_tol_vs_maxrank.pdf")

# =============================================================
# Individual figures — angular
# =============================================================
fig4, ax4 = subplots(1, 1, figsize=(5.5, 4.2))
for (i, nPN) in enumerate(nPN_list)
    ax4.loglog(ϑ_list, errors_ang[i, :], ang_markers[i] * "--",
               color=ang_colors[i], linewidth=1.6, markersize=6.5,
               markerfacecolor=ang_colors[i], markeredgecolor="white", markeredgewidth=0.6,
               label=L"$N_{PN} = $" * "$nPN")
end
ax4.set_xlabel(L"\vartheta");  ax4.set_ylabel(L"$\|\phi_r - \phi_{\mathrm{full}}\|_{L^2}$")
ax4.legend(loc="lower right", title=L"$N_{PN}$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
add_slope1!(ax4, ϑ_list)
polish_ax(ax4);  fig4.tight_layout()
savefig(joinpath(results_dir, "linesource_tol_vs_error_angular.pdf"))
println("Saved: Results/linesource_tol_vs_error_angular.pdf")

fig6, ax6 = subplots(1, 1, figsize=(5.5, 4.2))
for (i, nPN) in enumerate(nPN_list)
    ax6.semilogx(ϑ_list, max_ranks_ang[i, :], ang_markers[i] * "--",
                 color=ang_colors[i], linewidth=1.6, markersize=6.5,
                 markerfacecolor=ang_colors[i], markeredgecolor="white", markeredgewidth=0.6,
                 label=L"$N_{PN} = $" * "$nPN")
end
ax6.set_xlabel(L"\vartheta");  ax6.set_ylabel(L"r_{\max}")
ax6.legend(loc="upper right", title=L"$N_{PN}$", title_fontsize=12, fontsize=12, borderaxespad=0.4)
polish_ax(ax6);  fig6.tight_layout()
savefig(joinpath(results_dir, "linesource_tol_vs_maxrank_angular.pdf"))
println("Saved: Results/linesource_tol_vs_maxrank_angular.pdf")
