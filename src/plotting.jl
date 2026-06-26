function set_publication_style()
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
end

function plot_MLMCParams(Dict_levels::Dict,problem::String,ε::Float64)
    path = "Results/$problem/$ε"
    CheckCreateFolder(path);
    set_publication_style();

    β = Dict_levels["Convergence_rates"]["β"];
    γ = Dict_levels["Convergence_rates"]["γ"];

    L = length(Dict_levels)-2;
    variances_levels = zeros(Float64, L+1);
    costs_levels = zeros(Float64, L+1);
    N_samples_levels = zeros(Float64, L+1);
    for ℓ = 0:L
        variances_levels[ℓ+1] = Dict_levels["$ℓ"]["var"];
        costs_levels[ℓ+1] = Dict_levels["$ℓ"]["cost"];
        N_samples_levels[ℓ+1] = Dict_levels["$ℓ"]["N_samples"];
    end

    # --- Variance plot ---
    close("all");
    fig, ax = plt.subplots(figsize=(6, 5));
    V0 = variances_levels[1] / 2.0;
    ax.semilogy(0:L, variances_levels, label=L"V_\ell", marker="o", color="C0");
    ax.semilogy(0:L, [V0 * 2.0^(-β*ℓ) for ℓ = 0:L],
        label=latexstring("\\mathcal{O}(2^{-\\beta \\ell}),\\; \\beta = ", round(β, sigdigits=3)),
        linestyle="--", color="C3");
    ax.semilogy(0:L, [V0 * 2.0^(-ℓ) for ℓ = 0:L],
        label=L"\mathcal{O}(2^{-\ell})", linestyle="-.", color="C4");
    ax.semilogy(0:L, [V0 * 2.0^(-2ℓ) for ℓ = 0:L],
        label=L"\mathcal{O}(2^{-2\ell})", linestyle=":", color="C2");
    ax.set_xticks(0:L);
    ax.set_xlabel(L"Level $\ell$", fontsize=25);
    ax.set_ylabel(L"Variance $V_\ell$", fontsize=25);
    ax.tick_params(labelsize=20);
    ax.legend(fontsize=22);
    ax.minorticks_on();
    ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
    plt.tight_layout();
    plt.savefig("$path/MLMC_Variances_levels.pdf", bbox_inches="tight");

    # --- Cost plot ---
    close("all");
    fig, ax = plt.subplots(figsize=(6, 5));
    C0 = 2.0 * costs_levels[1];
    ax.semilogy(0:L, costs_levels, label=L"C_\ell", marker="s", color="C0");
    ax.semilogy(0:L, [C0 * 2.0^(γ*ℓ) for ℓ = 0:L],
        label=latexstring("\\mathcal{O}(2^{\\gamma \\ell}),\\; \\gamma = ", round(γ, sigdigits=3)),
        linestyle="--", color="C3");
    ax.semilogy(0:L, [C0 * 2.0^(ℓ) for ℓ = 0:L],
        label=L"\mathcal{O}(2^{\ell})", linestyle="-.", color="C4");
    ax.semilogy(0:L, [C0 * 2.0^(2ℓ) for ℓ = 0:L],
        label=L"\mathcal{O}(2^{2\ell})", linestyle=":", color="C2");
    ax.set_xticks(0:L);
    ax.set_xlabel(L"Level $\ell$", fontsize=25);
    ax.set_ylabel(L"Cost $C_\ell$", fontsize=25);
    ax.tick_params(labelsize=20);
    ax.legend(fontsize=22);
    ax.minorticks_on();
    ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
    plt.tight_layout();
    plt.savefig("$path/MLMC_Costs_levels.pdf", bbox_inches="tight");

    # --- Number of samples plot ---
    close("all");
    fig, ax = plt.subplots(figsize=(6, 5));
    ax.semilogy(0:L, N_samples_levels, marker="D", color="C0");
    ax.set_xticks(0:L);
    ax.set_xlabel(L"Level $\ell$", fontsize=25);
    ax.set_ylabel(L"Number of samples $N_\ell$", fontsize=25);
    ax.tick_params(labelsize=20);
    ax.minorticks_on();
    ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
    plt.tight_layout();
    plt.savefig("$path/MLMC_N_samples_levels.pdf", bbox_inches="tight");
end

function plot_mean_DLR(obj::UQSetup,Dict_levels::Dict,L::Int)
    problem = obj.Problem_levels["0"]["DLRASetup"].solver.settings.problem;
    ε = obj.ε;
    path = "Results/$problem/$ε"
    CheckCreateFolder(path);
    if obj.Interpolate == "1D"
        settings_L = obj.Problem_levels["$L"]["DLRASetup"].solver.settings;
       
        if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            x_L = settings_L.xMid;
        else
            x_L = settings_L.x;
        end
        mean_L = zeros(Float64, length(x_L));
        for ℓ = 0:L
            Dict_levels["$ℓ"]["rank_diff"] = obj.Problem_levels["$ℓ"]["rank_diff"]/Dict_levels["$ℓ"]["N_samples"];
            settings_ℓ = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings;
            if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                x_ℓ = settings_ℓ.xMid;
            else
                x_ℓ = settings_ℓ.x;
            end
            mean_ℓ_interp = linear_interpolation(x_ℓ, Dict_levels["$ℓ"]["mean"][1]);
            mean_L .+= mean_ℓ_interp(x_L);
        end

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        ax.plot(x_L, mean_L, label="DLR-MLMC", linewidth=2.0);

        if haskey(obj.Problem_levels,"ref")
            x_ref = obj.Problem_levels["ref"]["x"];
            mean_ref = obj.Problem_levels["ref"]["mean"];
            ax.plot(x_ref, mean_ref, label="Reference", linestyle="--", linewidth=2.0);
        end

        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$\langle \phi \rangle (x,t)$");
        ax.legend();
        ax.minorticks_on();
        ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_ExpectedVal_$(obj.label)_$L.pdf", bbox_inches="tight");
    elseif obj.Interpolate == "2D"
        mean_L = zeros(length(Dict_levels["$L"]["mean"][1]));
        x_fine = obj.Problem_levels["$L"]["DLRASetup"].solver.settings.xMid;
        y_fine = obj.Problem_levels["$L"]["DLRASetup"].solver.settings.yMid;
        nx = obj.Problem_levels["$L"]["DLRASetup"].solver.settings.NCellsX;
        ny = obj.Problem_levels["$L"]["DLRASetup"].solver.settings.NCellsY;

        fineGrid = RectangleGrid(x_fine,y_fine);
        
        for ℓ = 0:L
            x_coarse = obj.Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.xMid;
            y_coarse = obj.Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.yMid;
            coarseGrid = RectangleGrid(x_coarse,y_coarse);
            # Dict_levels["$ℓ"]["rank_diff"] = obj.Problem_levels["$ℓ"]["rank_diff"]/Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ = Dict_levels["$ℓ"]["mean"][1];
            y = zeros(nx,ny);

            for i = 1:nx
                for j = 1:ny
                    z = fineGrid[i,j];
                    grid_val = GridInterpolations.interpolate(coarseGrid,mean_ℓ,z);
                    y[i,j] = grid_val;
                end
            end
            mean_L .+= vec(y);
        end
        u1 = Vec2Mat(nx,ny,mean_L)
        X = (x_fine[2:end-1]'.*ones(size(x_fine[2:end-1])));
        Y = (y_fine[2:end-1]'.*ones(size(y_fine[2:end-1])))';

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        if problem == "Lattice"
            norm = PyPlot.matplotlib.colors.LogNorm(vmin=1e-8, vmax=4)
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        elseif problem == "Hohlraum"
            norm = PyPlot.matplotlib.colors.Normalize(vmin=0, vmax=5)
            data = u1[2:(end-1),(end-1):-1:2]
        else
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        end
        pcm = ax.pcolormesh(X, Y, data, shading="auto", rasterized=true, cmap="inferno", norm=norm);
        cbar = fig.colorbar(pcm, ax=ax);
        cbar.ax.tick_params(labelsize=11);
        if problem == "Hohlraum"
            cbar.set_label(L"$\langle \phi \rangle$", fontsize=13)
        end
        ax.set_aspect("equal");
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$y$");
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_ExpectedVal_$(obj.label)_$L.pdf", bbox_inches="tight");
    else
        Throw(ArgumentError("Plotting has not be coded yet"))
    end
end

function plot_mean(obj::UQSetup,Dict_levels::Dict,L::Int)
    ε = obj.ε;
    problem = obj.Problem_levels["0"]["solver"].settings.problem;
    path = "Results/$problem/$ε"
    CheckCreateFolder(path);
    if obj.Interpolate == "1D"
        settings_L = obj.Problem_levels["$L"]["solver"].settings;
        if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            x_L = settings_L.xMid;
        else
            x_L = settings_L.x;
        end
        mean = zeros(Float64, length(x_L));
        
        for ℓ = 0:L
            Dict_levels["$ℓ"]["rank_diff"] = obj.Problem_levels["$ℓ"]["rank_diff"]/Dict_levels["$ℓ"]["N_samples"];
            settings_ℓ = obj.Problem_levels["$ℓ"]["solver"].settings;
            if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                x_ℓ = settings_ℓ.xMid;
            else
                x_ℓ = settings_ℓ.x;
            end
            mean_ℓ_interp = linear_interpolation(x_ℓ, Dict_levels["$ℓ"]["mean"][1], extrapolation_bc=Line());
            mean .+= mean_ℓ_interp(x_L);
        end
        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        ax.plot(x_L, mean, label="DLR-MLMC", linewidth=2.0);
        if haskey(obj.Problem_levels,"ref")
            x_ref = obj.Problem_levels["ref"]["x"];
            mean_ref = obj.Problem_levels["ref"]["mean"];
            ax.plot(x_ref, mean_ref, label="Reference", linestyle="--", linewidth=2.0);
        end
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$Q_{$ℓ}$");
        ax.legend();
        ax.minorticks_on();
        ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_ExpectedVal_$(obj.label)_$L.pdf", bbox_inches="tight");
    elseif obj.Interpolate == "2D"
        mean = zeros(length(Dict_levels["$L"]["mean"][1]));
        x_fine = obj.Problem_levels["$L"]["solver"].settings.xMid;
        y_fine = obj.Problem_levels["$L"]["solver"].settings.yMid;
        nx = obj.Problem_levels["$L"]["solver"].settings.NCellsX;
        ny = obj.Problem_levels["$L"]["solver"].settings.NCellsY;

        fineGrid = RectangleGrid(x_fine,y_fine);
        
        for ℓ = 0:L
            x_coarse = obj.Problem_levels["$(ℓ)"]["solver"].settings.xMid;
            y_coarse = obj.Problem_levels["$(ℓ)"]["solver"].settings.yMid;
            coarseGrid = RectangleGrid(x_coarse,y_coarse);
            # Dict_levels["$ℓ"]["rank_diff"] = obj.Problem_levels["$ℓ"]["rank_diff"]/Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ = Dict_levels["$ℓ"]["mean"][1];
            y = zeros(nx,ny);

            for i = 1:nx
                for j = 1:ny
                    z = fineGrid[i,j];
                    grid_val = GridInterpolations.interpolate(coarseGrid,mean_ℓ,z);
                    y[i,j] = grid_val;
                end
            end
            mean .+= vec(y);
        end
        u1 = Vec2Mat(nx,ny,mean)
        X = (x_fine[2:end-1]'.*ones(size(x_fine[2:end-1])));
        Y = (y_fine[2:end-1]'.*ones(size(y_fine[2:end-1])))';

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        if problem == "Lattice"
            norm = PyPlot.matplotlib.colors.LogNorm(vmin=1e-8, vmax=4)
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        elseif problem == "Hohlraum"
            norm = PyPlot.matplotlib.colors.Normalize(vmin=0, vmax=5)
            data = u1[2:(end-1),(end-1):-1:2]
        else
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        end
        pcm = ax.pcolormesh(X, Y, data, shading="auto", rasterized=true, cmap="inferno", norm=norm);
        cbar = fig.colorbar(pcm, ax=ax);
        cbar.ax.tick_params(labelsize=11);
        if problem == "Hohlraum"
            cbar.set_label(L"$Q_{$ℓ}$", fontsize=13)
        end
        ax.set_aspect("equal");
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$y$");
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_ExpectedVal_$(obj.label)_$L.pdf", bbox_inches="tight");
    else
        Throw(ArgumentError("Plotting has not be coded yet"))
    end
end

function plot_dQ(obj::UQSetup,Dict_levels::Dict,ℓ::Int)
    ε = obj.ε;
    problem = obj.Problem_levels["0"]["solver"].settings.problem;
    path = "Results/$problem/$ε"
    CheckCreateFolder(path);
    if obj.Interpolate == "1D"
        settings_ℓ = obj.Problem_levels["$ℓ"]["solver"].settings;
        if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            x_ℓ = settings_ℓ.xMid;
        else
            x_ℓ = settings_ℓ.x;
        end
        mean_ℓ =  Dict_levels["$ℓ"]["mean"][1];

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        ax.plot(x_ℓ, mean_ℓ, linewidth=2.0);
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(latexstring("\\Delta Q_{$ℓ}"));
        ax.minorticks_on();
        ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_DeltaQ_$(obj.label)_$ℓ.pdf", bbox_inches="tight");
    elseif obj.Interpolate == "2D"
        mean = Dict_levels["$ℓ"]["mean"][1];
        x_fine = obj.Problem_levels["$ℓ"]["solver"].settings.xMid;
        y_fine = obj.Problem_levels["$ℓ"]["solver"].settings.yMid;
        nx = obj.Problem_levels["$ℓ"]["solver"].settings.NCellsX;
        ny = obj.Problem_levels["$ℓ"]["solver"].settings.NCellsY;

        u1 = Vec2Mat(nx,ny,Dict_levels["$ℓ"]["mean"][1])
        X1 = (x_fine[2:end-1]'.*ones(size(x_fine[2:end-1])));
        Y1 = (y_fine[2:end-1]'.*ones(size(y_fine[2:end-1])))';

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        if problem == "Lattice"
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        elseif problem == "Hohlraum"
            norm = PyPlot.matplotlib.colors.Normalize(vmin=0, vmax=5)
            data = u1[2:(end-1),(end-1):-1:2]'
        else
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        end
        pcm = ax.pcolormesh(X1, Y1, data, shading="auto", rasterized=true, cmap="inferno", norm=norm);
        cbar = fig.colorbar(pcm, ax=ax);
        cbar.ax.tick_params(labelsize=11);
        if problem == "Hohlraum"
            cbar.set_label(L"$\Delta Q_\ell \; [\phi]$", fontsize=13)
        end
        ax.set_aspect("equal");
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$y$");
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_DeltaQ_$(obj.label)_$ℓ.pdf", bbox_inches="tight");
    else
        Throw(ArgumentError("Plotting has not be coded yet"))
    end
   return nothing;
end

function plot_dQ_DLR(obj::UQSetup,Dict_levels::Dict,ℓ::Int)
    problem = obj.Problem_levels["0"]["DLRASetup"].solver.settings.problem;
    ε = obj.ε;
    path = "Results/$problem/$ε"
    CheckCreateFolder(path);
    if obj.Interpolate == "1D"
        settings_ℓ = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings;
        if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            x_ℓ = settings_ℓ.xMid;
        else
            x_ℓ = settings_ℓ.x;
        end
        mean_ℓ =  Dict_levels["$ℓ"]["mean"][1];

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        ax.plot(x_ℓ, mean_ℓ, linewidth=2.0);
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(latexstring("\\Delta Q_{$ℓ}"));
        ax.minorticks_on();
        ax.grid(true, which="major", linestyle="-", linewidth=0.5, alpha=0.3);
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_DeltaQ_$(obj.label)_$ℓ.pdf", bbox_inches="tight");
    elseif obj.Interpolate == "2D"
        mean = Dict_levels["$ℓ"]["mean"][1];
        x_fine = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings.xMid;
        y_fine = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings.yMid;
        nx = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings.NCellsX;
        ny = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings.NCellsY;

        u1 = Vec2Mat(nx,ny,Dict_levels["$ℓ"]["mean"][1])
        X1 = (x_fine[2:end-1]'.*ones(size(x_fine[2:end-1])));
        Y1 = (y_fine[2:end-1]'.*ones(size(y_fine[2:end-1])))';

        close("all");
        set_publication_style();
        fig, ax = plt.subplots(figsize=(6, 5));
        if problem == "Lattice"
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        elseif problem == "Hohlraum"
            norm = PyPlot.matplotlib.colors.Normalize() # vmin=0, vmax=5
            data = u1[2:(end-1),(end-1):-1:2]
        else
            norm = PyPlot.matplotlib.colors.Normalize()
            data = 4.0*pi*u1[2:(end-1),(end-1):-1:2]'
        end
        pcm = ax.pcolormesh(X1, Y1, data, shading="auto", rasterized=true, cmap="inferno", norm=norm);
        cbar = fig.colorbar(pcm, ax=ax);
        cbar.ax.tick_params(labelsize=11);
        if problem == "Hohlraum"
            cbar.set_label(L"$\Delta Q_\ell \; [\phi]$", fontsize=13)
        end
        ax.set_aspect("equal");
        ax.set_xlabel(L"$x$");
        ax.set_ylabel(L"$y$");
        plt.tight_layout();
        plt.savefig("Results/$problem/$ε/MLMC_DeltaQ_$(obj.label)_$ℓ.pdf", bbox_inches="tight");
    else
        Throw(ArgumentError("Plotting has not be coded yet"))
    end
   return nothing;
end


function CheckCreateFolder(path::String)
    if isdir(path)
    else
        mkpath(path);
    end
end