function plot_MLMCParams(Dict_levels::Dict, problem::String, ε::Float64, maxlevel::Int)
    path = "Results/$problem/$ε"
    CheckCreateFolder(path)
    β = Dict_levels["Convergence_rates"]["β"]
    γ = Dict_levels["Convergence_rates"]["γ"]

    L = min(maxlevel, length(Dict_levels) - 2)
    variances_levels = zeros(Float64, L + 1)
    costs_levels     = zeros(Float64, L + 1)
    N_samples_levels = zeros(Float64, L + 1)
    for ℓ = 0:L
        variances_levels[ℓ+1] = Dict_levels["$ℓ"]["var"]
        costs_levels[ℓ+1]     = Dict_levels["$ℓ"]["cost"]
        N_samples_levels[ℓ+1] = Dict_levels["$ℓ"]["N_samples"]
    end

    levels = 0:L
    V0 = variances_levels[1]
    C0 = costs_levels[1]

    # ── Variances ──────────────────────────────────────────────────────────────
    close("all")
    fig, ax = plt.subplots(figsize=(5, 4))  # smaller, like the target

    ax.semilogy(levels, variances_levels,
                label=L"V_\ell",
                marker="s", markersize=6, color="steelblue", linewidth=1.5)
    ax.semilogy(levels, [V0 * 2.0^(-β * ℓ) for ℓ in levels],
                label=string(L"\mathcal{O}(2^{-\beta\ell})", ", β = ", round(β, sigdigits=3)),
                linestyle="--", color="red", linewidth=1.2)
    ax.semilogy(levels, [V0 * 2.0^(-ℓ) for ℓ in levels],
                label=L"\mathcal{O}(2^{-\ell})",
                linestyle="-.", color="purple", linewidth=1.2)
    ax.semilogy(levels, [V0 * 2.0^(-2ℓ) for ℓ in levels],
                label=L"\mathcal{O}(2^{-2\ell})",
                linestyle=":", color="green", linewidth=1.2)

    ax.set_xticks(collect(levels))
    ax.set_xticklabels(["$ℓ" for ℓ in levels])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(L"Level $\ell$", fontsize=11)
    ax.set_ylabel(L"Variance $V_\ell$", fontsize=11)
    ax.tick_params(axis="both", labelsize=8)
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("$path/MLMC_Variances_levels.png", bbox_inches="tight", dpi=150)

    # ── Costs ──────────────────────────────────────────────────────────────────
    close("all")
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.semilogy(levels, costs_levels,
                label=L"C_\ell",
                marker="s", markersize=6, color="steelblue", linewidth=1.5)
    ax.semilogy(levels, [C0 * 2.0^(γ * ℓ) for ℓ in levels],
                label=string(L"\mathcal{O}(2^{\gamma\ell})", ", γ = ", round(γ, sigdigits=3)),
                linestyle="--", color="red", linewidth=1.2)
    ax.semilogy(levels, [C0 * 2.0^(ℓ) for ℓ in levels],
                label=L"\mathcal{O}(2^{\ell})",
                linestyle="-.", color="purple", linewidth=1.2)
    ax.semilogy(levels, [C0 * 2.0^(2ℓ) for ℓ in levels],
                label=L"\mathcal{O}(2^{2\ell})",
                linestyle=":", color="green", linewidth=1.2)

    ax.set_xticks(collect(levels))
    ax.set_xticklabels(["$ℓ" for ℓ in levels])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel(L"Level $\ell$", fontsize=11)
    ax.set_ylabel(L"Cost $C_\ell$", fontsize=11)
    ax.tick_params(axis="both", labelsize=8)
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("$path/MLMC_Costs_levels.png", bbox_inches="tight", dpi=150)

    # ── Sample counts ──────────────────────────────────────────────────────────
    close("all")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(levels, N_samples_levels, marker="s", markersize=6, color="steelblue")
    ax.set_xticks(collect(levels))
    ax.set_xticklabels(["$ℓ" for ℓ in levels])
    ax.set_xlabel(L"Level $\ell$", fontsize=11)
    ax.set_ylabel("No. of samples", fontsize=11)
    ax.tick_params(axis="both", labelsize=8)
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("$path/MLMC_N_samples_levels.png", bbox_inches="tight", dpi=150)
end

function _save_3D_panels(mean3d::AbstractArray{<:Real,3},
                         density::AbstractArray{<:Real,3},
                         settings, path::String, tag::String;
                         vmax=nothing, is_difference=false)

    if is_difference
        vmax_val     = isnothing(vmax) ? maximum(abs.(mean3d)) : Float64(vmax)
        vmin_val     = -vmax_val
        cmap         = "managua_r"
        alpha_thresh = 0.025 * vmax_val
        alpha_fn_XZ  = m -> clamp.(abs.(m) ./ alpha_thresh, 0.0, 1.0)'
        alpha_fn_XY  = m -> clamp.(abs.(m) ./ alpha_thresh, 0.0, 1.0)
    else
        vmax_val = isnothing(vmax) ? maximum(mean3d) : Float64(vmax)
        vmin_val = 0.0
        cmap     = "jet"
        alpha_thresh = 0.025 * vmax_val
        alpha_fn_XZ = m -> Float64.(m .> alpha_thresh)'
        alpha_fn_XY = m -> Float64.(0.5 .* m .> alpha_thresh)
    end

    idxX = Int(ceil(0.225*settings.NCellsX ))
    idxZ = Int(ceil(settings.NCellsZ / 2))

    YZ  = (settings.yMid' .* ones(size(settings.zMid)))'
    Z   = (settings.zMid'  .* ones(size(settings.yMid)))
    X   = (settings.xMid'  .* ones(size(settings.yMid)))
    XY  = (settings.yMid'  .* ones(size(settings.xMid)))'

    # ── XZ slice (coronal) ────────────────────────────────────────────────────
    close("all")
    y_extent = settings.yMid[end] - settings.yMid[1]
    z_extent = settings.zMid[end] - settings.zMid[1]
    fig, ax = plt.subplots(figsize=(10, 10 * z_extent / y_extent))
    ax.pcolormesh(YZ, Z, density[idxX, :, :],
                  vmin=0.0, vmax=1.85, cmap="gray")
    pcm = ax.pcolormesh(YZ, Z, mean3d[idxX, :, :],
                        alpha=alpha_fn_XZ(mean3d[idxX, :, :]'),
                        vmin=vmin_val, vmax=vmax_val, cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    #cbar.set_label(is_difference ? "Dose difference" : "Dose", fontsize=16)
    ax.grid(linestyle="dotted")
    ax.set_xlabel(L"y", fontsize=14); ax.set_ylabel(L"z", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    #ax.set_title(tag * " – XZ slice", fontsize=25)
    plt.savefig("$path/$(tag)_XZ.png", bbox_inches="tight")

    # ── XY slice (axial) ──────────────────────────────────────────────────────
    close("all")
    x_extent = settings.xMid[end] - settings.xMid[1]
    y_extent = settings.yMid[end] - settings.yMid[1]
    fig, ax = plt.subplots(figsize=(10, 10 * y_extent / x_extent))
    ax.pcolormesh(XY, X, density[:, :, idxZ]', cmap="gray")
    pcm = ax.pcolormesh(XY, X, mean3d[:, :, idxZ]',
                        alpha=alpha_fn_XY(mean3d[:, :, idxZ]'),
                        vmin=vmin_val, vmax=vmax_val, cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    #cbar.set_label(is_difference ? "Dose difference" : "Dose", fontsize=16)
    ax.grid(linestyle="dotted")
    ax.set_xlabel(L"x", fontsize=14); ax.set_ylabel(L"y", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    plt.savefig("$path/$(tag)_XY.png", bbox_inches="tight")
end

function CheckCreateFolder(path::String)
    if !isdir(path)
        mkpath(path)
    end
end