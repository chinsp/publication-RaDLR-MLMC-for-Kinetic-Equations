__precompile__
using Random
using Parameters
using JLD2
using Interpolations
using LinearRegression
using Distributed


mutable struct UQSetup{T<:AbstractFloat}
    ε::T
    N_samples::Int
    Numerical_solver::Function
    pdf
    # Sol_to_FoI::Function
    # L::Int # Number of levels for fixed level MLMC
    FoI::String
    Problem_levels::Dict
    UQ_solver::Function
    budget::Array{T,1};

    function UQSetup(N_samples::Int,Numerical_solver_name::String,Problem_levels::Dict,pdf,FoI::String,UQ_solver_name::String,ε::T,budget::Array{T,1}=[10,36000,90])
        if Problem_levels == Dict()
            println("Problem levels not set up yet")
            throw(ArgumentError("Problem level for ℓ=0 must be set up before creating the SetupUQ object."))
            return nothing;
        end

        ## If the method is adaptive, N_samples is the number of warm-up samples.
        if UQ_solver_name == "MC"
        elseif UQ_solver_name == "MLMC_adaptive"
            UQ_solver = MLMC_adaptive;
        elseif UQ_solver_name == "MLMC_fixedLevels"
            UQ_solver = MLMC_fixedLevels;
        elseif UQ_solver_name == "MLMC_adaptive_threaded"
            UQ_solver = MLMC_adaptive_threaded;
        else
            println("UQ solver not implemented yet")
            throw(ArgumentError("UQ solver not implemented yet."))
        end

        if Numerical_solver_name == "csdAugBUG"
            Numerical_solver = TITUS.Solve_sample; 
        else
            println("Numerical solver not implemented yet")
            throw(ArgumentError("Numerical solver not implemented yet."))
        end

        return new{T}(ε,N_samples,Numerical_solver,pdf,FoI,Problem_levels,UQ_solver,budget)
    end
end

function run(obj::UQSetup)
    Dict_levels = obj.UQ_solver(obj);
    return Dict_levels;
end

function Sol_to_FoI(g,FoI::String) #Solution to funciton of interest mapping
    if FoI == "ScalarFlux"
        # m,n = size(g);
        # e1 = zeros(n);
        # e1[1] = 1.0;
        y = g[:,1];
    elseif FoI == "FullSolution"
        y = g;  
    end
    return y;
end

function generateSamples(samples::Array,level::Int)
    epsilon = 2.0^(-level);

    if level == 0    
        sample_sum = sum(samples .* (1+epsilon));
        sample_sqsum = sum((samples.*(1+epsilon)).^2);
    else
        sample_sum = sum(samples .*(1+epsilon) .- samples .*(1+2*epsilon));
        sample_sqsum = sum((samples.*(1+epsilon) .- samples .*(1+2*epsilon)).^2);
    end
    return sample_sum, sample_sqsum;
end

function check_level_setup(obj::UQSetup, ℓ::Int)
    # Check if the level ℓ is already set up in Problem_levels
    problem = obj.Problem_levels["0"]["settings"].problem;
    if haskey(obj.Problem_levels, "$ℓ")
        
    else
        if haskey(obj.Problem_levels, "$(ℓ-1)")
            if problem == "SingleBeam" || problem == "BoxInsert" || problem == "smallCT" 
                Nx_new = (obj.Problem_levels["$(ℓ-1)"]["settings"].Nx-3)*2 + 3;
                Ny_new = (obj.Problem_levels["$(ℓ-1)"]["settings"].Ny-3)*2 + 3;
                Nz_new = (obj.Problem_levels["$(ℓ-1)"]["settings"].Nz-3)*2 + 3;
                write_config(
                particle=obj.Problem_levels["$(ℓ-1)"]["settings"].particle,
                problem=obj.Problem_levels["$(ℓ-1)"]["settings"].problem,
                model=obj.Problem_levels["$(ℓ-1)"]["settings"].model,
                OmegaMin=0,
                eKin=obj.Problem_levels["$(ℓ-1)"]["settings"].mu_e,
                nx=Nx_new,
                ny=Ny_new,
                nz=Nz_new,
                nMoments=obj.Problem_levels["$(ℓ-1)"]["settings"].nPN,
                order=2,
                rank=obj.Problem_levels["$(ℓ-1)"]["settings"].r,
                maxRank=obj.Problem_levels["$(ℓ-1)"]["settings"].rMax,
                solverName="",
                tracerFileName="",
                cfl=obj.Problem_levels["$(ℓ-1)"]["settings"].cfl,
                tolerance=obj.Problem_levels["$(ℓ-1)"]["settings"].epsAdapt,
                trace=true,
                disableGPU=false)

                settings = TITUS.Settings("configFiles/temp.toml");
                settings.epsAdapt = settings.dx#^2
                obj.Problem_levels["$ℓ"] = Dict(
                    "settings" => settings,
                    "solver" => TITUS.SolverGPU(settings));
            else
                println("Problem not coded yet")
            end
        else
            println("Problem level $(ℓ-1) not set up yet")
        end
    end
    return nothing;
end

function ComputeDiff(Qf_ℓ0::Array{T,1}, dQ::Array{T,1}, Problem_levels::Dict, ℓ::Int) where {T<:AbstractFloat}
    if ℓ == 0
        return dQ
    else 
        Qf_ℓ0 = reshape(Qf_ℓ0, Problem_levels["$(ℓ-1)"]["settings"].NCellsX, Problem_levels["$(ℓ-1)"]["settings"].NCellsY, Problem_levels["$(ℓ-1)"]["settings"].NCellsZ)

        x_range = range(Problem_levels["$(ℓ-1)"]["settings"].xMid[1], Problem_levels["$(ℓ-1)"]["settings"].xMid[end], length=length(Problem_levels["$(ℓ-1)"]["settings"].xMid))
        y_range = range(Problem_levels["$(ℓ-1)"]["settings"].yMid[1], Problem_levels["$(ℓ-1)"]["settings"].yMid[end], length=length(Problem_levels["$(ℓ-1)"]["settings"].yMid))
        z_range = range(Problem_levels["$(ℓ-1)"]["settings"].zMid[1], Problem_levels["$(ℓ-1)"]["settings"].zMid[end], length=length(Problem_levels["$(ℓ-1)"]["settings"].zMid))

        x_fine = round.(Problem_levels["$(ℓ)"]["settings"].xMid, digits=4)
        y_fine = round.(Problem_levels["$(ℓ)"]["settings"].yMid, digits=4)
        z_fine = round.(Problem_levels["$(ℓ)"]["settings"].zMid, digits=4)

        itp = interpolate(Qf_ℓ0, BSpline(Cubic(Interpolations.Line(OnGrid()))))
        sitp = Interpolations.scale(itp, x_range, y_range, z_range)
        etp = extrapolate(sitp, 0.0)

        dQ .-= etp(x_fine, y_fine, z_fine)[:]
        return dQ
    end
end

function ComputeDiff(Qf_ℓ0::T,dQ::T,Problem_levels::Dict,ℓ::Int) where {T<:AbstractFloat}
    # If the FoI is a scalar, the QoI is simply the difference between the two levels.
    if ℓ == 0
        return dQ;
    else 
        return dQ - Qf_ℓ0;
    end
end

function ComputeSample(obj,ℓ::Int,sample::Array{T,1}) where {T<:AbstractFloat}

    #Nx_ℓ1 = obj.Problem_levels["$ℓ"]["settings"].Nx;
    #Qf_ℓ1 = zeros(T,Nx_ℓ1); #dont preallocate memory
    solver_ℓ1 = obj.Problem_levels["$ℓ"]["solver"];
    println(T)
    Qf_ℓ1 = Sol_to_FoI(obj.Numerical_solver(solver_ℓ1,Float32.(sample)),obj.FoI) #[Int((length(obj.x)+1)/2)];
    dQ = Qf_ℓ1;
    if ℓ != 0
        Nx_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["settings"].Nx;
        solver_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"];
        Qf_ℓ0 = Sol_to_FoI(obj.Numerical_solver(solver_ℓ0,Float32.(sample)),obj.FoI);
        dQ = ComputeDiff(Qf_ℓ0, dQ, Problem_levels, ℓ);
    else
        solver_ℓ0 = obj.Problem_levels["$(ℓ)"]["solver"];
        Qf_ℓ0 = Sol_to_FoI(obj.Numerical_solver(solver_ℓ0,Float32.(sample)),obj.FoI);
        dQ = Qf_ℓ0;
    end

    return dQ,Qf_ℓ1;
end

function MLMC_fixedLevels(obj::UQSetup)
    # Estimating variances, costs, and sample sizes for the two-level Monte Carlo method    
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    Nx0 = obj.Problem_levels["0"]["settings"].Nx-1; # Number of points on the coarsest level
    println("Provide the maximum levels for the MLMC method (L):");
    L = readline(); # Number of levels
    if L == ""
        L = 5; # Default value
    elseif L == "kill"
        println("Exiting the MLMC_fixedLevels method.")
        return;
    else
        L = parse(Int, L);
    end
    println("Number of levels: ", L)
    for ℓ = 0:L
        check_level_setup(obj, ℓ);
    end
    Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(T, obj.Problem_levels["$ℓ"]["settings"].Nx), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);

    α, β = 0.0, 0.0;

    time_start = time();
    @sync begin
        for ℓ = 0:L
            N_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            Nx = length(Dict_levels["$ℓ"]["mean"]);
            dx = obj.Problem_levels["$ℓ"]["settings"].dx;
        
            
            count = Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ =  Dict_levels["$ℓ"]["mean"];
            delta_ℓ = zeros(T,length(Dict_levels["$ℓ"]["mean"]));
            var_ℓ = Dict_levels["$ℓ"]["var"];
            
            time = 0.0;
            time_list = [];
            Threads.@threads for n = 1:N_ℓ
                count += 1;
                
                alpha = [rand(pdf[i]) for i = 1:length(pdf)];

                t = @elapsed begin
                    dQ,Qf = ComputeSample(obj,ℓ,alpha);
                end
                time_list = push!(time_list, t);
                if n > 6
                    time += t;
                end
                if ℓ == 0
                    delta_ℓ .= (Qf - mean_ℓ);
                    mean_ℓ .+= delta_ℓ./count;
                    var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
                else
                    delta_ℓ .= (dQ - mean_ℓ);
                    mean_ℓ .+= delta_ℓ./count;
                    var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
                end
            end
            println(ℓ)
            N_ℓ = Dict_levels["$ℓ"]["N_samples"] + N_ℓ;
            Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
            var_ℓ = var_ℓ*dx/(N_ℓ - 1);


            Dict_levels["$ℓ"]["mean"] = mean_ℓ;
            Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
            Dict_levels["$ℓ"]["cost"] = minimum(time_list);
            Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
            # println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
            # println("------------------------------------------------------------------")
        end
    end

    for ℓ = 2:L
        Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
        Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
    end

    println("Estimating the rates α, β, and γ from the variances and costs")
    if α
    else
        α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
    end
    # α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
    # α = 1.0;
    β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
    lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
    γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
    cγ = 2^(LinearRegression.bias(lr_γ)[1])
    println("α: ", α)
    println("β: ", β)
    println("γ: ", γ)
    println("cγ: ", cγ)
    println("------------------------------------------------------------------")

    println("Estimating the optimal number of samples for each level")
    println("------------------------------------------------------------------")

    Nsamples_new = zeros(Int, L);
    for ℓ = 0:L
        Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
        if Dict_levels["$ℓ"]["Add_samples"] > 0
            println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
        end
    end 
    println("------------------------------------------------------------------")
    @sync begin
        for ℓ = 0:L
            N_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            Nx = length(Dict_levels["$ℓ"]["mean"]);
            dx = obj.Problem_levels["$ℓ"]["settings"].dx;
            
            count = Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ =  Dict_levels["$ℓ"]["mean"];
            delta_ℓ = zeros(T,length(Dict_levels["$ℓ"]["mean"]));
            var_ℓ = Dict_levels["$ℓ"]["var"];
            
            
            time = 0.0;
            if N_ℓ == 0
                # println("No additional samples needed on level $(ℓ-1)")
                N_ℓ = Nsamples[ℓ];
                continue
            else
                # println("Computing additional samples on level $(ℓ-1)")
                Threads.@threads for n = 1:N_ℓ
                    count += 1;
                    alpha = [rand(pdf[i]) for i = 1:length(pdf)];
                    if ℓ == 2
                        println("Level: ", ℓ, ", Sample: ", n, ", alpha: ", alpha)
                    end

                    # obj.alpha = alpha;
                    dQ,Qf = ComputeSample(obj,ℓ,alpha);
                    if (ℓ - 1) == 0
                        delta_ℓ = (Qf - mean_ℓ);
                        mean_ℓ .+= delta_ℓ./count;
                        var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
                    else
                        delta_ℓ = (dQ - mean_ℓ);
                        mean_ℓ .+= delta_ℓ./count;
                        var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
                    end
                end
            end
            Dict_levels["$ℓ"]["N_samples"] += N_ℓ;
            var_ℓ = var_ℓ*dx/(N_ℓ - 1);

            Dict_levels["$ℓ"]["mean"] = mean_ℓ;
            Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
            Dict_levels["$ℓ"]["cost"] = minimum(time_list);
            Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
            # println("------------------------------------------------------------------")
        end
    end

   total_var = 0.0;
    for ℓ = 0:L
        total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
    end

    time_end = time();
    println("Total time taken: ", (time_end - time_start), " seconds")

    total_cost = 0.0;
    for ℓ = 0:L
        total_cost += costs_levels[ℓ]*Nsamples_new[ℓ];
    end

    settings_L = obj.Problem_levels["$L"]["settings"];
    mean = zeros(T, settings_L.Nx);
    for ℓ = 0:L
        settings_ℓ = obj.Problem_levels["$ℓ"]["settings"];
        mean_ℓ_interp = linear_interpolation(settings_ℓ.x, Dict_levels["$ℓ"]["mean"]);
        mean .+= mean_ℓ_interp(settings_L.x);
    end
    close("all");
    plt.figure(figsize=(10, 10));
    plt.plot(settings_L.x,mean,label=string("Mean solution at t=",settings_L.Tend));
    plt.grid(linestyle="dotted");
    plt.xlabel(L"x");
    plt.ylabel(L"u(x,t)");
    plt.title(string("Mean solution at t = ",settings_L.Tend));
    plt.savefig("MLMC_ExpectedVal.png",bbox_inches="tight");
    
    # plot_MLMCParams(Dict_levels);

    return Dict_levels;
end

function MLMC_adaptive_step(obj,Dict_levels,ℓ)
    dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
    Nx = length(Dict_levels["$ℓ"]["mean"]);
    dx = obj.Problem_levels["$ℓ"]["settings"].dx;
    pdf = obj.pdf;
    
    count = Dict_levels["$ℓ"]["N_samples"];
    mean_ℓ =  Dict_levels["$ℓ"]["mean"];
    delta_ℓ = zeros(T,length(Dict_levels["$ℓ"]["mean"]));
    var_ℓ = Dict_levels["$ℓ"]["var"];

    time = 0.0;
    if dN_ℓ >0
        
        Threads.@threads for n = 1:dN_ℓ
            count += 1;
            alpha = [rand(pdf[i]) for i = 1:length(pdf)];

            t = @elapsed begin
            dQ,Qf = ComputeSample(obj,ℓ,alpha);
            end
            time += t;
            if ℓ == 0
                delta_ℓ .= (Qf - mean_ℓ);
                mean_ℓ .+= delta_ℓ./count;
                var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
            else
                delta_ℓ .= (dQ - mean_ℓ);
                mean_ℓ .+= delta_ℓ./count;
                var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
            end
        end
        N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
        Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
        var_ℓ = var_ℓ*dx/(N_ℓ - 1);


        Dict_levels["$ℓ"]["mean"] = mean_ℓ;
        Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
        Dict_levels["$ℓ"]["cost"] = time/N_ℓ;
        Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
        println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", time/N_ℓ)
        println("------------------------------------------------------------------")
        flush(stdout)
    end
    return Dict_levels;
end

function MLMC_adaptive(obj::UQSetup)
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    Nx0 = obj.Problem_levels["0"]["settings"].Nx-1; # Number of points on the coarsest level
    L = 2; # Number of levels
    for ℓ = 0:L
        check_level_setup(obj, ℓ);
    end
    println("Checked level set-up")
    flush(stdout)
    Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(T, prod(obj.Problem_levels["$ℓ"]["settings"].gridSize)), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    α, β = 0.0, 0.0;
  
    time_start = time();
    while sum([Dict_levels["$ℓ"]["Add_samples"] for ℓ = 0:L]) > 0
        # global α, β
        for ℓ = 0:L
            CUDA.reclaim()
            println("Starting computation for initial samples, level $ℓ")
            flush(stdout)
            #Random.seed!(1234)
            dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            Nx = length(Dict_levels["$ℓ"]["mean"]);
            dx = prod(obj.Problem_levels["$ℓ"]["settings"].gridWidth);
            
            count = Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ =  Dict_levels["$ℓ"]["mean"];
            delta_ℓ = zeros(T,length(Dict_levels["$ℓ"]["mean"]));
            var_ℓ = Dict_levels["$ℓ"]["var"];

            time = 0.0;
            time_list = [];
            if dN_ℓ >0
                for n = 1:dN_ℓ
                    println("Sample $n/$dN_ℓ")
                    flush(stdout)
                    count += 1;
                    alpha = [rand(pdf[i]) for i = 1:length(pdf)];
                    println("alpha = $alpha")
                    t = @elapsed begin
                    dQ,Qf = ComputeSample(obj,ℓ,alpha);
                    end
                    time_list = push!(time_list, t);
                    if n > 6
                        time += t;
                    end
                    if ℓ == 0
                        delta_ℓ .= (Qf - mean_ℓ);
                        mean_ℓ .+= delta_ℓ./count;
                        var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
                    else
                        delta_ℓ .= (dQ - mean_ℓ);
                        mean_ℓ .+= delta_ℓ./count;
                        var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
                    end
                end
                N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
                Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
                var_ℓ = var_ℓ*dx/(N_ℓ - 1);
    
    
                Dict_levels["$ℓ"]["mean"] = mean_ℓ;
                Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
                Dict_levels["$ℓ"]["cost"] = minimum(time_list);
                Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
                println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
                println("------------------------------------------------------------------")
                flush(stdout)
            end
    

            
        end

        for ℓ = 2:L
            Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
            Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
        end
    
        println("Estimating the rates α, β, and γ from the variances and costs")
        flush(stdout)
        if α != 0.0
        else
            α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
        end
        # α = 1.0;
        β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
        lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
        γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
        cγ = 2^(LinearRegression.bias(lr_γ)[1])
        println("α: ", α)
        println("β: ", β)
        println("γ: ", γ)
        println("cγ: ", cγ)
        println("------------------------------------------------------------------")
        # Dict_levels["$(L+1)"] = Dict("mean" => zeros(T, Nx0 * 2^(L+1)), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples)

    
        println("Estimating the optimal number of samples for each level")
        println("------------------------------------------------------------------")
        Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
        time_end = time();
        println("Total time taken: ", (time_end - time_start), " seconds")
        total_var = 0.0;
        for ℓ = 0:L
            total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
        end
        rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
        println("Bias: ", rem)
        println("Total variance: ", total_var)
        flush(stdout)
        settings_L = obj.Problem_levels["$L"]["settings"];
        mean_total = zeros(T, prod(settings_L.gridSize));
        x_fine = round.(Problem_levels["$(L)"]["settings"].xMid,digits=4);
        y_fine = round.(Problem_levels["$(L)"]["settings"].yMid,digits=4);
        z_fine = round.(Problem_levels["$(L)"]["settings"].zMid,digits=4);
        for ℓ = 0:L
            settings_ℓ = obj.Problem_levels["$ℓ"]["settings"]
            mean_ℓ = reshape(Dict_levels["$ℓ"]["mean"], settings_ℓ.NCellsX, settings_ℓ.NCellsY, settings_ℓ.NCellsZ)
            save("output/mean_total_$ℓ.jld", "mean_total", mean_ℓ)

            x_range = range(settings_ℓ.xMid[1], settings_ℓ.xMid[end], length=length(settings_ℓ.xMid))
            y_range = range(settings_ℓ.yMid[1], settings_ℓ.yMid[end], length=length(settings_ℓ.yMid))
            z_range = range(settings_ℓ.zMid[1], settings_ℓ.zMid[end], length=length(settings_ℓ.zMid))

            itp = interpolate(mean_ℓ, BSpline(Cubic(Interpolations.Line(OnGrid()))))
            sitp = Interpolations.scale(itp, x_range, y_range, z_range)
            etp = extrapolate(sitp, 0.0)

            mean_total .+= etp(x_fine, y_fine, z_fine)[:]
        end
        mean_total = reshape(mean_total,settings_L.NCellsX ,settings_L.NCellsY ,settings_L.NCellsZ)
        
        save("output/mean_total.jld", "mean_total", mean_total)   
        save("output/Dict_levels.jld", "dict_levels", Dict_levels)
            
        flush(stdout)
        for ℓ = 0:L
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
            if Dict_levels["$ℓ"]["Add_samples"] > 0
                println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                flush(stdout)
            end
        end 
        println("Checking additional costs...")
        costs_addSamples = estimateCosts(Dict_levels,L)
        println("Additional samples expected to take $costs_addSamples seconds")
        
        CUDA.reclaim()
        free, total = CUDA.memory_info()
        if sum([Dict_levels["$ℓ"]["Add_samples"]>0.01*Dict_levels["$ℓ"]["N_samples"] for ℓ = 0:L]) == 0 
            rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
            # println(rem,",", ε/sqrt(2))
            if rem > ε/sqrt(2)
                println("Adding a new level")
                L = L + 1;
                check_level_setup(obj, L);
                Dict_levels["$L"] = Dict("mean" => zeros(T, Int(prod(obj.Problem_levels["$L"]["settings"].gridSize))), "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L) , "N_samples" => 0, "Add_samples" => 0);
                for ℓ = 0:L
                    Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]+1);
                    if Dict_levels["$ℓ"]["Add_samples"] > 0
                        println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                    end
                end 
                println("Checking additional costs...")
                costs_addSamples = estimateCosts(Dict_levels,L)
                println("Additional samples expected to take $costs_addSamples seconds")
                if L >= obj.budget[1] || ((time() - time_start) + costs_addSamples) > obj.budget[2] || (total-free)/total*100 > obj.budget[3] #option to stop computation according to level, runtime or used GPU memory
                    println("Stopped because budget limit reached at Level $L, expected time after add. samples $(((time() - time_start) + costs_addSamples)) or GPU memory $((total-free)/total*100)%")
                    flush(stdout)
                    Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                    time_end = time();
                    println("Total time taken: ", (time_end - time_start), " seconds")
                    total_var = 0.0;
                    for ℓ = 0:L
                        total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                    end
                    println("Bias: ", rem)
                    println("Total variance: ", total_var)
                    flush(stdout)
                    settings_L = obj.Problem_levels["$L"]["settings"];
                    mean_total = zeros(T, prod(settings_L.gridSize));
                    x_fine = round.(Problem_levels["$(L)"]["settings"].xMid,digits=4);
                    y_fine = round.(Problem_levels["$(L)"]["settings"].yMid,digits=4);
                    z_fine = round.(Problem_levels["$(L)"]["settings"].zMid,digits=4);
                    for ℓ = 0:L
                        settings_ℓ = obj.Problem_levels["$ℓ"]["settings"]
                        mean_ℓ = reshape(Dict_levels["$ℓ"]["mean"], settings_ℓ.NCellsX, settings_ℓ.NCellsY, settings_ℓ.NCellsZ)
                        save("output/mean_total_$ℓ.jld", "mean_total", mean_ℓ)

                        x_range = range(settings_ℓ.xMid[1], settings_ℓ.xMid[end], length=length(settings_ℓ.xMid))
                        y_range = range(settings_ℓ.yMid[1], settings_ℓ.yMid[end], length=length(settings_ℓ.yMid))
                        z_range = range(settings_ℓ.zMid[1], settings_ℓ.zMid[end], length=length(settings_ℓ.zMid))

                        itp = interpolate(mean_ℓ, BSpline(Cubic(Interpolations.Line(OnGrid()))))
                        sitp = Interpolations.scale(itp, x_range, y_range, z_range)
                        etp = extrapolate(sitp, 0.0)

                        mean_total .+= etp(x_fine, y_fine, z_fine)[:]
                    end
                    mean_total = reshape(mean_total,settings_L.NCellsX ,settings_L.NCellsY ,settings_L.NCellsZ)
                    
                    save("output/mean_total.jld", "mean_total", mean_total)  
                    break    
                end
            else
                println("Converged")
                Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                time_end = time();
                println("Total time taken: ", (time_end - time_start), " seconds")
                total_var = 0.0;
                for ℓ = 0:L
                    total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                end
                println("Bias: ", rem)
                println("Total variance: ", total_var)
                settings_L = obj.Problem_levels["$L"]["settings"];
                mean_total = zeros(T, prod(settings_L.gridSize));
                x_fine = round.(Problem_levels["$(L)"]["settings"].xMid,digits=4);
                y_fine = round.(Problem_levels["$(L)"]["settings"].yMid,digits=4);
                z_fine = round.(Problem_levels["$(L)"]["settings"].zMid,digits=4);
                for ℓ = 0:L
                    settings_ℓ = obj.Problem_levels["$ℓ"]["settings"]
                    mean_ℓ = reshape(Dict_levels["$ℓ"]["mean"], settings_ℓ.NCellsX, settings_ℓ.NCellsY, settings_ℓ.NCellsZ)
                    save("output/mean_total_$ℓ.jld", "mean_total", mean_ℓ)

                    x_range = range(settings_ℓ.xMid[1], settings_ℓ.xMid[end], length=length(settings_ℓ.xMid))
                    y_range = range(settings_ℓ.yMid[1], settings_ℓ.yMid[end], length=length(settings_ℓ.yMid))
                    z_range = range(settings_ℓ.zMid[1], settings_ℓ.zMid[end], length=length(settings_ℓ.zMid))

                    itp = interpolate(mean_ℓ, BSpline(Cubic(Interpolations.Line(OnGrid()))))
                    sitp = Interpolations.scale(itp, x_range, y_range, z_range)
                    etp = extrapolate(sitp, 0.0)

                    mean_total .+= etp(x_fine, y_fine, z_fine)[:]
                end
                mean_total = reshape(mean_total,settings_L.NCellsX ,settings_L.NCellsY ,settings_L.NCellsZ)
                
                idxX = Int(ceil(settings_L.NCellsX/2))
                Z = (settings_L.zMid'.*ones(size(settings_L.yMid)))
                YZ = (settings_L.yMid'.*ones(size(settings_L.zMid)))'

                close("all");
                plt.figure(figsize=(10, 10));
                plt.pcolormesh(YZ',Z',mean_total[idxX,:,:],vmin=0,vmax=maximum(mean_total[idxX,:,:]),cmap="jet");
                plt.grid(linestyle="dotted");
                plt.title(string("Mean solution"));
                plt.savefig("Results/MLMC_ExpectedVal_$(settings_L.problem)_$(settings_L.particle).png",bbox_inches="tight");     
                break
            end
        else
            if L >= obj.budget[1] || ((time() - time_start) + costs_addSamples) > obj.budget[2] || (total-free)/total*100 > obj.budget[3] #option to stop computation according to level, runtime or used GPU memory
                    rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
                    println("Stopped because budget limit reached at Level $L, expected time after add. samples $(((time() - time_start)+ costs_addSamples)) or GPU memory $((total-free)/total*100)%")
                    flush(stdout)
                    Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                    time_end = time();
                    println("Total time taken: ", (time_end - time_start), " seconds")
                    total_var = 0.0;
                    for ℓ = 0:L
                        total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                    end
                    println("Bias: ", rem)
                    println("Total variance: ", total_var)
                    flush(stdout)
                    settings_L = obj.Problem_levels["$L"]["settings"];
                    mean_total = zeros(T, prod(settings_L.gridSize));
                    x_fine = round.(Problem_levels["$(L)"]["settings"].xMid,digits=4);
                    y_fine = round.(Problem_levels["$(L)"]["settings"].yMid,digits=4);
                    z_fine = round.(Problem_levels["$(L)"]["settings"].zMid,digits=4);
                    for ℓ = 0:L
                        settings_ℓ = obj.Problem_levels["$ℓ"]["settings"]
                        mean_ℓ = reshape(Dict_levels["$ℓ"]["mean"], settings_ℓ.NCellsX, settings_ℓ.NCellsY, settings_ℓ.NCellsZ)
                        save("output/mean_total_$ℓ.jld", "mean_total", mean_ℓ)

                        x_range = range(settings_ℓ.xMid[1], settings_ℓ.xMid[end], length=length(settings_ℓ.xMid))
                        y_range = range(settings_ℓ.yMid[1], settings_ℓ.yMid[end], length=length(settings_ℓ.yMid))
                        z_range = range(settings_ℓ.zMid[1], settings_ℓ.zMid[end], length=length(settings_ℓ.zMid))

                        itp = interpolate(mean_ℓ, BSpline(Cubic(Interpolations.Line(OnGrid()))))
                        sitp = Interpolations.scale(itp, x_range, y_range, z_range)
                        etp = extrapolate(sitp, 0.0)

                        mean_total .+= etp(x_fine, y_fine, z_fine)[:]
                    end
                    mean_total = reshape(mean_total,settings_L.NCellsX ,settings_L.NCellsY ,settings_L.NCellsZ)
                    
                    save("output/mean_total.jld", "mean_total", mean_total)       
                    break
            end
        end
    end
    return Dict_levels;
end

function MLMC_adaptive_threaded(obj::UQSetup)

    if Threads.nthreads() == 1
        return throw(ArgumentError("MLMC_adaptive_threaded requires multiple threads. Please run with multiple threads enabled."));
    end

    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    Nx0 = obj.Problem_levels["0"]["settings"].Nx-1; # Number of points on the coarsest level
    L = 2; # Number of levels
    for ℓ = 0:L
        check_level_setup(obj, ℓ);
    end
    Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(T, obj.Problem_levels["$ℓ"]["settings"].Nx), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    α, β = 0.0, 0.0;

    time_start = time();
    while sum([Dict_levels["$ℓ"]["Add_samples"] for ℓ = 0:L]) > 0
        # global α, β
        @sync begin
            for ℓ = 0:L
                MLMC_adaptive_step(obj,Dict_levels,ℓ);
            end
        end
       

        for ℓ = 2:L
            Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
            Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
        end
    
        println("Estimating the rates α, β, and γ from the variances and costs")
        if α
        else
            α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
        end
        β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
        lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
        γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
        cγ = 2^(LinearRegression.bias(lr_γ)[1])
        println("α: ", α)
        println("β: ", β)
        println("γ: ", γ)
        println("cγ: ", cγ)
        println("------------------------------------------------------------------")

    
        println("Estimating the optimal number of samples for each level")
        println("------------------------------------------------------------------")
        for ℓ = 0:L
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
            if Dict_levels["$ℓ"]["Add_samples"] > 0
                println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
            end
        end 

        if sum([Dict_levels["$ℓ"]["Add_samples"]>0.01*Dict_levels["$ℓ"]["N_samples"] for ℓ = 0:L]) == 0 
            rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
            # println(rem,",", ε/sqrt(2))
            if rem > ε/sqrt(2)
                println("Adding a new level")
                L = L + 1;
                Dict_levels["$L"] = Dict("mean" => zeros(T, Nx0 * 2^(L) +1), "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L) , "N_samples" => 0, "Add_samples" => 0);
                for ℓ = 0:L
                    Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"] + 1 );
                    if Dict_levels["$ℓ"]["Add_samples"] > 0
                        println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                    end
                end 
            else
                println("Converged")
                Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                time_end = time();
                println("Total time taken: ", (time_end - time_start), " seconds")
                total_var = 0.0;
                for ℓ = 0:L
                    total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                end
                println("Bias: ", rem)
                println("Total variance: ", total_var)
                settings_L = obj.Problem_levels["$L"]["settings"];
                mean = zeros(T, settings_L.Nx);
                for ℓ = 0:L
                    settings_ℓ = obj.Problem_levels["$ℓ"]["settings"];
                    mean_ℓ_interp = linear_interpolation(settings_ℓ.x, Dict_levels["$ℓ"]["mean"]);
                    mean .+= mean_ℓ_interp(settings_L.x);
                end
                close("all");
                plt.figure(figsize=(10, 10));
                plt.plot(settings_L.x,mean,label=string("Mean solution at t=",settings_L.Tend));
                plt.grid(linestyle="dotted");
                plt.xlabel(L"x");
                plt.ylabel(L"u(x,t)");
                plt.title(string("Mean solution at t = ",settings_L.Tend));
                plt.savefig("MLMC_ExpectedVal.png",bbox_inches="tight");
            end
        end
    end
    return Dict_levels;
end

function estimateCosts(Dict_levels,L)
    costs = 0.0
    for ℓ = 0:L
        costs += Dict_levels["$ℓ"]["Add_samples"]*Dict_levels["$ℓ"]["cost"] #this tends to underestimate costs for new levels (especially at start of comp.)
    end 
    return costs
end

function write_config(;
    particle,
    problem,
    model,
    OmegaMin,
    eKin,
    nx,
    ny,
    nz,
    nMoments,
    order,
    rank,
    maxRank,
    solverName="",
    tracerFileName="",
    cfl,
    tolerance,
    trace,
    disableGPU,
    filepath="configFiles/temp.toml"
)
    content = """
[physics]
particle = "$particle"
problem = "$problem"
model = "$model"
OmegaMin = $OmegaMin
eKin = $eKin

[numerics]
nx = $nx
ny = $ny
nz = $nz
nMoments = $nMoments
order = $order
rank = $rank
maxRank = $maxRank
solverName = "$solverName"
tracerFileName = "$tracerFileName"
cfl = $cfl
tolerance = $tolerance

[computation]
trace = $(trace ? "true" : "false")
disableGPU = $(disableGPU ? "true" : "false")

[planning]

[refinement]
"""

    open(filepath, "w") do io
        write(io, content)
    end
end