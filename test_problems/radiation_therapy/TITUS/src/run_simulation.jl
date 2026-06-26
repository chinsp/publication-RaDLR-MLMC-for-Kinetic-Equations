function runAndPlot(file_path,write_files=false,plot_files=true)

    config = TOML.parsefile(file_path)
    trace = get(config["computation"], "trace", "false")
    disableGPU = get(config["computation"], "disableGPU", "false")
    mu_e = get(config["physics"], "eKin", 90)
    collided_model = get(config["physics"], "model", "Boltzmann")
    file_name = split(file_path, ".")[1]
    order = get(config["numerics"], "order", 2)

    close("all")
    
    info = "CUDA"
    @timeit to "Geometry and physics set-up" begin
        s = Settings(file_path);
        if CUDA.functional() && ~disableGPU
            solver1 = SolverGPU(s,order);
        else
            solver1 = SolverCPU(s,order);
        end
    end
    println("rMax = $(s.rMax)")
    println("r = $(s.r)")
    @timeit to "Solver" begin
        X_dlr,S_dlr,W_dlr,_, dose_DLR, dose_coll, rankInTime₂, ψ₂ = getfield(Main,Symbol("Solve$(s.solverName)"))(solver1,s.model,trace,collided_model);
    end
    println("Average rank = $(mean(rankInTime₂[2,2:end]))")
    u = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,X_dlr*Diagonal(S_dlr)*W_dlr[1,:]);
    dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);
    dose_coll = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_coll);

    println(to)
    idxX = Int(ceil(s.NCellsX/2))
    idxY = Int(ceil(s.NCellsY/2))   
    idxZ = Int(ceil(s.NCellsZ/2))

    X = (s.xMid'.*ones(size(s.yMid)))
    Y = (s.yMid'.*ones(size(s.xMid)))'
    Z = (s.zMid'.*ones(size(s.yMid)))
    XZ = (s.xMid'.*ones(size(s.zMid)))
    ZX = (s.zMid'.*ones(size(s.xMid)))
    YZ = (s.yMid'.*ones(size(s.zMid)))'
   
    root = pkgdir(TITUS)
    #information for subfolder name
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    outdir = joinpath(root, "output", "run_" * timestamp)
    mkpath(outdir)

    # write vtk file
    vtkfile = vtk_grid(joinpath(outdir,"dose_nPN$(s.nPN)"), s.xMid, s.yMid,s.zMid)
    vtkfile["dose"] = dose_DLR
    vtkfile["dose_normalized"] = dose_DLR./sum(dose_DLR[dose_DLR.>0]) * sum(mu_e)
    vtkfile["dose_uncollided"] = dose_DLR .- dose_coll
    vtkfile["dose_collided"] = dose_coll
    vtkfile["densityHU"] = s.densityHU
    outfiles = vtk_save(vtkfile)

    if write_files 
        file_name = "rankInEnergy_nPN$(s.nPN)_tol$(s.epsAdapt)_$(s.tracerFileName).jld2"
        save(joinpath(outdir, file_name), "energy", solver1.csd.eGrid[2:end], "rank", rankInTime₂[2,:])
        file_name = "dose_nPN$(s.nPN)_tol$(s.epsAdapt)_$(s.tracerFileName).jld2"
        save(joinpath(outdir, file_name), "dose", dose_DLR./sum(dose_DLR[dose_DLR.>0]) * sum(mu_e),"x", s.xMid,"y",s.yMid,"z",s.zMid)
    end 

    if plot_files
        Omegas = [s.Omega1 s.Omega2 s.Omega3]  # or however many beams

        function best_slice_axis(Omegas)
            # unit basis vectors
            axes = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]

            # score each axis by total alignment with beams
            scores = zeros(3)

            for (axis_idx, e) in enumerate(axes)
                for j in 1:size(Omegas, 1)
                    Ω = Omegas[j, :]
                    scores[axis_idx] += abs(dot(normalize(Ω), e))
                end
            end

            # choose axis with minimal alignment
            return argmin(scores)
        end

        function slice_along_axis(dose, axis_s, idxX,idxY,idxZ)
            if axis_s == 1
                return dose[idxX, :, :]
            elseif axis_s == 2
                return dose[:, idxY, :]
            else
                return dose[:, :, idxZ]
            end
        end

        function best_depth_axis(Omegas)
            axes = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]

            scores = zeros(3)

            for (axis_idx, e) in enumerate(axes)
                for j in 1:size(Omegas, 1)
                    Ω = Omegas[j, :]
                    scores[axis_idx] += abs(dot(normalize(Ω), e))
                end
            end

            # choose axis with maximum alignment
            return argmax(scores)
        end
        function line_along_axis(dose, axis_s, i, j, k, z)
            if axis_s == 1
                return z, dose[:, j, k]
            elseif axis_s == 2
                return z, dose[i, :, k]
            else
                return z, dose[i, j, :]
            end
        end
        slice_axis = best_slice_axis(Omegas)
        depth_axis = best_depth_axis(Omegas)

        slice_dlr = slice_along_axis(dose_DLR, slice_axis,idxX,idxY,idxZ)
        slice_dens = slice_along_axis(s.densityHU, slice_axis,idxX,idxY,idxZ)

        z, line_dlr = line_along_axis(dose_DLR, depth_axis, idxX, idxY, idxZ, s.zMid)
        z, line_coll = line_along_axis(dose_DLR, depth_axis, idxX, idxY, idxZ, s.zMid)
        line_uncoll = line_dlr - line_coll

        #plot ranks
        epsAdapt=s.epsAdapt
        fig = figure()
        ax = gca()
        ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
        labelvec = ["1st order","2nd order",L"\vartheta = $s.epsAdapt"]
        ax.plot(rankInTime₂[1,2:end].-s.eRest,rankInTime₂[2,2:end], "-g", label="ϑ=$epsAdapt",linewidth=2, alpha=1.0)
        ax.set_xlabel("pseudo time", fontsize=20);
        ax.set_ylabel("rank", fontsize=20);
        ax.tick_params("both",labelsize=20) 
        ax.legend(loc="upper left", fontsize=20)
        fig.canvas.draw() # Update the figure
        savefig(joinpath(root, "output/ranks_$(s.tracerFileName)_tol$(s.epsAdapt).png"))

        # Assume dose_DLR is (20, 20, 70)
        nx, ny, nz = size(dose_DLR)

        # Normalize data to [0,1]
        dmin, dmax = extrema(dose_DLR)
        norm_dose = (dose_DLR .- dmin) ./ (dmax - dmin)

        fig = figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_box_aspect((nx, ny, nz))
        threshold = 0.1   # below this → transparent
        function meshgrid(x, y)
           X = repeat(x', length(y), 1)
           Y = repeat(y, 1, length(x))
           return X, Y
        end
        for k in 1:nz
            slice = norm_dose[:, :, k]

            # Mask low values
            masked = copy(slice)
            masked[masked .< threshold] .= NaN

            # Create coordinate grid
            x = 1:nx
            y = 1:ny
            X, Y = meshgrid(x, y)
            cmap = PyPlot.cm.jet

            # Nonlinear alpha mapping (tune gamma)
            gamma = 2.5

            facecolors = Array{NTuple{4, Float64}}(undef, size(masked))

            for i in eachindex(masked)
                if isnan(masked[i])
                    facecolors[i] = (0.0, 0.0, 0.0, 0.0)  # fully transparent
                else
                    val = masked[i]

                    # RGB color from colormap
                    r, g, b, _ = cmap(val)

                    # Alpha mapped nonlinearly to dose
                    α = val^gamma

                    facecolors[i] = (r, g, b, α)
                end
            end
            # Plot as a surface (flat plane at z = k)
           ax.plot_surface(
                X, Y, fill(k, size(X)),
                facecolors = facecolors,
                rstride = 1, cstride = 1,
                shade = false
            )
        end

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        savefig(joinpath(outdir, "dose_3d.png"))

        # normalize dose
        dose_max = maximum(slice_dlr)
        dose_norm = slice_dlr ./ dose_max

        # alpha mapping
        threshold = 0.05
        γ = 0.4
        alpha_map = clamp.((dose_norm .- threshold) ./ (1 - threshold), 0, 1)
        alpha_map = alpha_map .^ γ
        alpha_map = alpha_map .* (dose_norm .> threshold)

        # plot
        fig = figure(figsize=(size(slice_dlr,1)/10,size(slice_dlr,2)/10))
        ax = gca()
        # plot
        ax.pcolormesh(slice_dens', cmap="Greys", shading="auto")

        im1 = ax.pcolormesh(
            slice_dlr',
            vmin=0,
            vmax=dose_max,
            cmap="jet",
            shading="auto",
            alpha=alpha_map'   # <-- key line
        )

        savefig(joinpath(outdir, "dose_slice.png"))

        fig = figure(dpi=200)
        ax = gca()
        ax.plot(z,line_dlr,label="dose")
        ax.plot(z,line_coll, label="dose collided")
        ax.plot(z,line_uncoll,label="dose uncollided")
        ax.legend()
        savefig(joinpath(outdir, "dose_lineCut.png"))

    end
            
    # # # #plot angular basis
    # file_names = glob("Wdlr_$(s.tracerFileName)*.txt")
    # for i=1:length(file_names)
    #     file_name_Wdlr = split.(file_names[i], ".")[1]
    #     run(`python plotW_new.py "$file_name_Wdlr"`)
    # end

    println("main for config $(file_path) finished")
    return dose_DLR;
end
