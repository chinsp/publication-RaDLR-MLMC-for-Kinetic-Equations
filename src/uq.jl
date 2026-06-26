__precompile__
using Random
using Parameters
using JLD2
using Interpolations
using GridInterpolations
using LinearRegression
using Distributed
using CUDA

mutable struct UQSetup
    ε::Float64 # the RMSE of any adaptive method
    N_samples::Int

    λ::Float64 # Mixing parameter for control variates

    Numerical_solver::Function
    pdf
    
    FoI::Vector{String}
    
    Problem_levels::Dict
    
    UQ_solver::Function

    Interpolate::String

    Plot::Array{Bool,1};
    Verbose::Bool

    label::String # Label for experiment name

    function UQSetup(Numerical_solver_name::String,Problem_levels::Dict,pdf,FoI::Union{String,Vector{String}},UQ_solver_name::String,ε::Float64,Interpolate::String,N_samples::Int=20,λ::Float64=0.0)
        label = "test"
        FoI = isa(FoI, String) ? [FoI] : FoI;
        if haskey(Problem_levels["0"],"DLRASetup")
            problem = Problem_levels["0"]["DLRASetup"].solver.settings.problem;
        else
            problem = Problem_levels["0"]["solver"].settings.problem;
        end

        if Problem_levels == Dict()
            println("Problem levels not set up yet")
            throw(ArgumentError("Problem level for ℓ=0 must be set up before creating the SetupUQ object."))
            return nothing;
        end

        ## If the method is adaptive, N_samples is the number of warm-up samples.
        if UQ_solver_name == "MC"
            UQ_solver = MC;
        elseif UQ_solver_name == "MC_adaptive"
            UQ_solver = MC_adaptive;
        elseif UQ_solver_name == "ControlVariate"
            UQ_solver = ControlVariate;
        elseif UQ_solver_name == "MLMC_adaptive"
            UQ_solver = MLMC_adaptive;
        elseif UQ_solver_name == "CUMLMC_adaptive"
            UQ_solver = CUMLMC_adaptive;
        elseif UQ_solver_name == "MLMC_fixedLevels"
            UQ_solver = MLMC_fixedLevels;
        elseif UQ_solver_name == "MLMC_adaptive_threaded"
            UQ_solver = MLMC_adaptive_threaded;
        else
            println("UQ solver not implemented yet")
            throw(ArgumentError("UQ solver not implemented yet."))
        end

        if problem == "Linesource" || problem == "Lattice" || problem == "Planesource" || problem == "PlanePulse" || problem == "GaussianPulse"
            if Numerical_solver_name == "Full"
                Numerical_solver = FullProblem;
                if haskey(Problem_levels["0"],"DLRASetup")
                    Numerical_solver = FullProblem;
                else
                    Numerical_solver = FullProblem;
                end
            elseif Numerical_solver_name == "frBUG"
                Numerical_solver = fixedrankBUG;
            elseif Numerical_solver_name == "fixedAugBUG"
                Numerical_solver = fixedaugBUG;
            elseif Numerical_solver_name == "AugBUG"
                if haskey(Problem_levels["0"],"DLRASetup")
                    Numerical_solver = augBUG;
                else
                    Numerical_solver = solveAugBUG;
                end
            elseif Numerical_solver_name == "ParBUG"
                Numerical_solver = parBUG;
            elseif Numerical_solver_name == "ParBUG_rejection"
                Numerical_solver = parBUG_rejection;
            elseif Numerical_solver_name == "SOAugBUG"
                Numerical_solver = SOaugBUG;
            elseif Numerical_solver_name == "SOParBUG"
                Numerical_solver = SOparBUG;
            elseif Numerical_solver_name == "AugBUG_gpu"
                Numerical_solver = augBUG_gpu;
            else
                throw(ArgumentError("Numerical solver for the problem not implemented yet."))
            end
        elseif problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            if Numerical_solver_name == "full"
                Numerical_solver = FullProblem;
            elseif Numerical_solver_name == "frBUG"
                Numerical_solver = fixedrankBUG;
            elseif Numerical_solver_name == "AugBUG"
                Numerical_solver = augBUG;
            elseif Numerical_solver_name == "ParBUG"
                Numerical_solver = parBUG;
            else
                throw(ArgumentError("Numerical solver for the problem not implemented yet."))
            end
                # ADD SOLVERS USED FOR THE RADIATION THERAPY PROBLEMS HERE
        elseif problem == "Hohlraum"
            if Numerical_solver_name == "AugBUG_gpu"
                Numerical_solver = augBUG_gpu;
            elseif Numerical_solver_name == "AugBUG"
                Numerical_solver = augBUG;
            else
                throw(ArgumentError("Numerical solver for Hohlraum not implemented: $Numerical_solver_name"))
            end
        else
            # println("Problem not implemented yet")
            throw(ArgumentError("Problem not implemented yet."))

        end

        Plot = [false,false];
        Verbose = true;

        return new(ε,N_samples,λ,Numerical_solver,pdf,FoI,Problem_levels,UQ_solver,Interpolate,Plot,Verbose,label)
    end
end

function run(obj::UQSetup)
    Dict_levels = obj.UQ_solver(obj);
    return Dict_levels;
end


function MC(obj::UQSetup)
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    ℓ = 0; # MC is equivalent to level 0 MLMC with level 0 corresponding to the finest level. Keep this  ℓ = 0 since it changes how things are computed later

    if haskey(obj.Problem_levels["$ℓ"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples));
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples));
    end

    dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
    if haskey(obj.Problem_levels["0"],"DLRASetup")
        dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
    else
        dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
    end

    count = Dict_levels["$ℓ"]["N_samples"];
    mean_ℓ =  Dict_levels["$ℓ"]["mean"];
    delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
    var_ℓ = Dict_levels["$ℓ"]["var"];

    time = 0.0;
    time_list = [];
    for n = 1:dN_ℓ
        count += 1;
        sample = [rand(pdf[i]) for i = 1:length(pdf)];

        t = @elapsed begin # Maybe switch the if loops around
            if haskey(obj.Problem_levels["0"],"DLRASetup")
                dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
            else
                dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
            end
        end
        dQ = dQs[1]; Qf = Qfs[1];
        time_list = push!(time_list, t);
        if n > 6
            time += t;
        end

        delta_ℓ .= (Qf - mean_ℓ);
        mean_ℓ .+= delta_ℓ./count;
        var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
    end

    N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
    Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
    var_ℓ = var_ℓ*dx/(N_ℓ - 1);


    Dict_levels["$ℓ"]["mean"] = mean_ℓ;
    Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
    Dict_levels["$ℓ"]["cost"] = minimum(time_list);
    Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
    if obj.Verbose
        println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
        println("------------------------------------------------------------------")
    end
    if obj.Plot[1]
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            plot_mean_DLR(obj,Dict_levels,ℓ);
        else
            plot_mean(obj,Dict_levels,ℓ);
        end
    end
    return Dict_levels;
end

function MC_adaptive(obj::UQSetup)
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    ℓ = 0; # MC is equivalent to level 0 MLMC with level 0 corresponding to the finest level. Keep this  ℓ = 0 since it changes how things are computed later

    if haskey(obj.Problem_levels["$ℓ"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples));
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples));
    end
    if haskey(obj.Problem_levels["0"],"DLRASetup")
        dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
    else
        dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
    end
    var_ℓ = 1.0;
    while var_ℓ >= ε^2
        dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];    

        count = Dict_levels["$ℓ"]["N_samples"];
        mean_ℓ =  Dict_levels["$ℓ"]["mean"];
        delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
        var_ℓ = Dict_levels["$ℓ"]["var"];

        time = 0.0;
        time_list = [];

        for n = 1:dN_ℓ
            count += 1;
            sample = [rand(pdf[i]) for i = 1:length(pdf)];

            t = @elapsed begin # Maybe switch the if loops around
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
                else
                    dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
                end
            end
            dQ = dQs[1]; Qf = Qfs[1];
            time_list = push!(time_list, t);
            if n > 6
                time += t;
            end

            delta_ℓ .= (Qf - mean_ℓ);
            mean_ℓ .+= delta_ℓ./count;
            var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
        end

        N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
        Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
        var_ℓ = var_ℓ*dx/(N_ℓ - 1);

        Dict_levels["$ℓ"]["mean"] = mean_ℓ;
        Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
        Dict_levels["$ℓ"]["cost"] = minimum(time_list);
        Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
        if obj.Verbose
            println("Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
            println("------------------------------------------------------------------")
        end
        if var_ℓ >= ε^2
            if obj.Verbose
                println("Estimating the optimal number of samples to reach specified RMSE")
                println("------------------------------------------------------------------")
            end
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, var_ℓ/ε^2));
            if obj.Verbose
                println("Additional samples: ", Dict_levels["$ℓ"]["Add_samples"])
            end
        end
        
        if obj.Plot[1] 
            if haskey(obj.Problem_levels["0"],"DLRASetup")
                plot_mean_DLR(obj,Dict_levels,ℓ);
            else
                plot_mean(obj,Dict_levels,ℓ);
            end
        end
    end    
    return Dict_levels;
end

function Sol_to_FoI(g::CuArray{Float32,2},FoI::String) #Solution to funciton of interest mapping
    if FoI == "ScalarFlux"
        m,n = size(g);
        e1 = CUDA.zeros(Float32,n);
        e1[1] = 1.0f0;
        y = g*e1;
    elseif FoI == "FullSolution"
        y = g;  
    end
    return y;
end

function Sol_to_FoI(g::Vector{Float64}, FoI::String)
    # phi returned by augBUG_gpu(obj, sample) is already the scalar flux
    return g
end

# ── ThermalSolution dispatch (Hohlraum / Lattice GPU problems) ────────────
# Spatial field QoI returns the full phi vector.
# Scalar QoI returns a 1-element Vector{Float64} so that ComputeDiff_DLR
# can detect it and skip interpolation.
if @isdefined(ThermalSolution)
function Sol_to_FoI(sol::ThermalSolution, FoI::String)
    Nt = sol.Nt
    QoI = sol.QoI
    if FoI == "ScalarFlux"
        return sol.phi
    # ── Hohlraum scalar QoI (time-integrated) ─────────────────────────────
    elseif FoI == "Absorption_GB"
        return [sum(QoI["$k"]["2"]["Green_Blue_block"] for k in 1:Nt)]
    elseif FoI == "Absorption_R"
        return [sum(QoI["$k"]["2"]["Red_block"] for k in 1:Nt)]
    elseif FoI == "Absorption_K"
        return [sum(QoI["$k"]["2"]["Black_block"] for k in 1:Nt)]
    elseif FoI == "MeanBlockAbsorption"
        return [sum(QoI["$k"]["3"] for k in 1:Nt)]
    elseif FoI == "MeanLineAbsorption"
        return [sum(QoI["$k"]["5"] for k in 1:Nt)]
    # ── Lattice scalar QoI (already time-integrated cumulatively) ─────────
    elseif FoI == "Flux_1.5"
        return [QoI["$Nt"]["2"]["1.5"]]
    elseif FoI == "Flux_2.5"
        return [QoI["$Nt"]["2"]["2.5"]]
    elseif FoI == "Absorption_Blue"
        return [QoI["$Nt"]["4"]]
    # ── Shared ────────────────────────────────────────────────────────────
    elseif FoI == "TotalMass"
        # Hohlraum stores mass in key "7", Lattice in key "5"
        last = QoI["$Nt"]
        return [haskey(last, "7") ? last["7"] : last["5"]]
    else
        throw(ArgumentError("Unknown FoI for ThermalSolution: $FoI"))
    end
end
end  # @isdefined(ThermalSolution)

function Sol_to_FoI(g::Array{Float64,2},FoI::String) #Solution to funciton of interest mapping
    if FoI == "ScalarFlux"
        y = g[:,1];
    elseif FoI == "Momentum"
        y = g[:,2];
    elseif FoI == "Velocity"
        y = g[:,2] ./ g[:,1];
    elseif FoI == "FullSolution" || FoI == "Dose"
        y = g;
    end
    return y;
end

function ControlVariate(obj::UQSetup)
    λ = obj.λ;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    L = 1; # ℓ = 0 corresponds to the control variate and ℓ =1 corresponds to the difference between the quantity to be estimated and control variate
    
    if haskey(obj.Problem_levels,"0") && haskey(obj.Problem_levels,"1")
    else
        Throw(DomainError("The level 0 and level 1 need to be setup before starting the control variates computation"))
        return nothing;
    end


    if haskey(obj.Problem_levels["$ℓ"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    end

    if sum(Dict_levels["0"]["mean"]) == 0 # If the expected value of the control variate is not known then it must be computed 
        for ℓ = 0:L
            dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];  
            if haskey(obj.Problem_levels["$ℓ"],"DLRASetup")
                dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
            else
                dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
            end

            count = Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ =  Dict_levels["$ℓ"]["mean"];
            delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
            var_ℓ = Dict_levels["$ℓ"]["var"];
            
            time = 0.0;
            time_list = [];

            for n = 1:dN_ℓ
                count += 1;
                sample = [rand(pdf[i]) for i = 1:length(pdf)];

                t = @elapsed begin # Maybe switch the if loops around
                    if haskey(obj.Problem_levels["0"],"DLRASetup")
                        dQ,Qf = ComputeSample_DLR(obj,ℓ,sample,n);
                    else
                        dQ,Qf = ComputeSample(obj,ℓ,sample,n);
                    end
                end
                time_list = push!(time_list, t);
                if n > 6
                    time += t;
                end
            
                delta_ℓ .= (Qf - mean_ℓ);
                mean_ℓ .+= delta_ℓ./count;
                var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
            end

            N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
            Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
            var_ℓ = var_ℓ*dx/(N_ℓ - 1);

            Dict_levels["$ℓ"]["mean"] = mean_ℓ;
            Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
            Dict_levels["$ℓ"]["cost"] = minimum(time_list);
            Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
            println("Expected value of control variate computed","Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
            println("------------------------------------------------------------------")
        end
    else
        ℓ = 1;
        dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];  
        if haskey(obj.Problem_levels["$ℓ"],"DLRASetup")
            dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
        else
            dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
        end

        count = Dict_levels["$ℓ"]["N_samples"];
        mean_ℓ =  Dict_levels["$ℓ"]["mean"];
        delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
        var_ℓ = Dict_levels["$ℓ"]["var"];
        
        time = 0.0;
        time_list = [];

        for n = 1:dN_ℓ
            count += 1;
            sample = [rand(pdf[i]) for i = 1:length(pdf)];

            t = @elapsed begin # Maybe switch the if loops around
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
                else
                    dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
                end
            end
            dQ = dQs[1]; Qf = Qfs[1];
            time_list = push!(time_list, t);
            if n > 6
                time += t;
            end

            delta_ℓ .= (Qf - mean_ℓ);
            mean_ℓ .+= delta_ℓ./count;
            var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
        end

        N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
        Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
        var_ℓ = var_ℓ*dx/(N_ℓ - 1);

        Dict_levels["$ℓ"]["mean"] = mean_ℓ;
        Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
        Dict_levels["$ℓ"]["cost"] = minimum(time_list);
        Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
        println("Expected value of control variate computed","Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
        println("------------------------------------------------------------------")
    end

    return Dict_levels;
end

function ControlVariate(obj,N_samples_s::Int,N_samples_diff::Int,N_samples_opt::Int,seed::Int,uncertParam::String,numerSolver::String,r::Int,s::Int,theta::Float64,d,epsilon::Float64,optTheta::Bool=false;kwargs...)
    if Threads.nthreads() > 1
        println("Warning: Incremental (co)variance computation might be inaccurate on several threads!")
    end
    @unpack FoI = kwargs

    if r <= s
        println("s must be smaller than r")
        return
    end

    if numerSolver == "full"
        NumerSolver = solveFullProblem;
    elseif numerSolver == "frBUG"
        NumerSolver = solvefixedrankBUG;
    elseif numerSolver == "fixedAugBUG"
        NumerSolver = solvefixedAugBUG;
    elseif numerSolver == "AugBUG"
        NumerSolver = solveAugBUG;
    end

    samples_diff = rand(d, N_samples_diff)
    samples_s = zeros(N_samples_s)
    samples_s[1:N_samples_diff] = samples_diff
    sol = zeros(Float64, obj.settings.Nx, N_samples_s)

    obj.uncertParam = uncertParam
    # Variables to track number of processed samples
    count_diff = 0
    count_s = 0
    # Initialize means, variances, and covariance 
    mean_s = zeros(Float64, obj.settings.Nx)
    mean_r = zeros(Float64, obj.settings.Nx)
    mean_diff = zeros(Float64,obj.settings.Nx)
    var_s = zeros(Float64, obj.settings.Nx)
    var_r = zeros(Float64, obj.settings.Nx)
    cov_rs = zeros(Float64, obj.settings.Nx)

    yr_n = zeros(N_samples_s, obj.settings.Nx)
    ys_n = zeros(N_samples_s, obj.settings.Nx)    

    if N_samples_opt ==0
        #compute theta_opt and number of samples using N_samples_diff warmup samples
        for n = 1:N_samples_diff
            sample = samples_diff[n]
            obj.sample = sample
            if numerSolver == "AugBUG"
                obj.settings.epsAdapt = r
            else
                obj.settings.r = r
            end
            gr = NumerSolver(obj)
    
            if numerSolver == "AugBUG"
                obj.settings.epsAdapt = s
            else
                obj.settings.r = s
            end
            gs = NumerSolver(obj)

            yr = Sol_to_FoI(obj, gr, FoI)
            ys = Sol_to_FoI(obj, gs, FoI)

            yr_n[n,:] = yr
            ys_n[n,:] = ys
        end

        # Compute sample variances and covariance
        mean_r = Statistics.mean(yr_n[1:N_samples_diff,:],dims=1)
        mean_s = Statistics.mean(ys_n[1:N_samples_diff,:],dims=1)
        var_r_tmp  = zeros(Float64,obj.settings.Nx);
        var_s_tmp  = zeros(Float64,obj.settings.Nx);
        cov_rs_tmp  = zeros(Float64,obj.settings.Nx);

        factor = obj.settings.dx / (N_samples_diff - 1)

        # Compute deviations from the mean
        dev_r = yr_n[1:N_samples_diff,:] .- mean_r
        dev_s = ys_n[1:N_samples_diff,:] .- mean_s

        # Variance for r
        var_r_tmp = sum(dev_r.^2, dims=1) .* factor

        # Variance for s
        var_s_tmp = sum(dev_s.^2, dims=1) .* factor

        # Covariance between r and s
        cov_rs_tmp = sum(dev_r .* dev_s, dims=1) .* factor
        var_r_tmp = norm(var_r_tmp)
        var_s_tmp = norm(var_s_tmp)
        cov_rs_tmp = norm(cov_rs_tmp)
        rho_rs = cov_rs_tmp/(sqrt(var_r_tmp)*sqrt(var_s_tmp))

        # # Compute the optimal theta
        # theta = cov_rs_tmp/ var_s_tmp

        #compute number samples
        N_diffOpt = maximum([Int(ceil(var_r_tmp * (1-rho_rs^2)/epsilon^2)) 5])
    else
        N_diffOpt =N_samples_opt
    end
    println("Computing $N_diffOpt samples on the differences")

    if N_diffOpt > N_samples_diff
        samples_diff = rand(d, N_diffOpt-N_samples_diff)
        samples_s[N_samples_diff+1:N_diffOpt] = samples_diff
        samples_s[N_diffOpt+1:N_samples_s] = rand(d,N_samples_s-N_diffOpt)

        #compute rest of samples for difference 
        for n = 1:N_diffOpt-N_samples_diff
            sample = samples_diff[n]
            obj.sample = sample
            if numerSolver == "AugBUG"
                obj.settings.epsAdapt = r
            else
                obj.settings.r = r
            end
            gr = NumerSolver(obj)
    
            if numerSolver == "AugBUG"
                obj.settings.epsAdapt = s
            else
                obj.settings.r = s
            end
            gs = NumerSolver(obj)

            yr = Sol_to_FoI(obj, gr, FoI)
            ys = Sol_to_FoI(obj, gs, FoI)

            yr_n[N_samples_diff+n,:] = yr
            ys_n[N_samples_diff+n,:] = ys
        end
    else
        samples_s[N_samples_diff+1:N_samples_s] = rand(d,N_samples_s-N_samples_diff)
    end

    mean_r = Statistics.mean(yr_n[1:N_diffOpt,:],dims=1)
    mean_s = Statistics.mean(ys_n[1:N_diffOpt,:],dims=1)

    var_r  = zeros(Float64,obj.settings.Nx);
    var_s_tmp  = zeros(Float64,obj.settings.Nx);
    cov_rs  = zeros(Float64,obj.settings.Nx);

    factor = obj.settings.dx / (N_diffOpt - 1)

    # Compute deviations from the mean
    dev_r = yr_n[1:N_diffOpt,:] .- mean_r
    dev_s = ys_n[1:N_diffOpt,:] .- mean_s

    # Variance for r
    var_r = sum(dev_r.^2, dims=1) .* factor

    # Variance for s
    var_s_tmp = sum(dev_s.^2, dims=1) .* factor

    # Covariance between r and s
    cov_rs = sum(dev_r .* dev_s, dims=1) .* factor
    var_r = norm(var_r)
    var_s_tmp = norm(var_s_tmp)
    cov_rs = norm(cov_rs)

    if optTheta
        # Compute the optimal theta
        theta = cov_rs/ var_s_tmp
        if theta > 1
            println("Warning: Optimal theta is larger than 1, most likely due to inaccurate (co)variance estimate. Setting theta to 1 for computations.")
            theta = 1;
        end
    end
    println("Optimal theta: $theta")
    
    mean_diff = mean_r - theta*mean_s

    # Process the additional samples for the low-accuracy solver `s`
    for n = (N_diffOpt + 1):N_samples_s
        sample = samples_s[n]
        obj.sample = sample
        if numerSolver == "AugBUG"
            obj.settings.epsAdapt = s
        else
            obj.settings.r = s
        end
        gs = NumerSolver(obj)

        ys = Sol_to_FoI(obj, gs, FoI)
        ys_n[n,:] = ys
    end
    mean_s = Statistics.mean(ys_n,dims=1)
    var_s  = zeros(Float64,obj.settings.Nx);

    dev_s = ys_n .- mean_s
    factor = obj.settings.dx / (N_samples_s - 1)
    # Variance for s
    var_s = sum(dev_s.^2, dims=1) .* factor
    var_s = norm(var_s)

    # Final variance of the control variate estimator
    var = var_r - 2 * theta * cov_rs + theta^2 * var_s

    # Final estimate of the mean (control variate estimator)
    mean_sol = theta * mean_s + mean_diff
    
    output = Dict("N_samples_s" => N_samples_s, "N_samples_diff" => N_diffOpt, "mean" => mean_sol, "var" => var,"sample"=>theta);

    return output;
end

function ControlVariate(obj,N_samples_s::Int,N_samples_diff::Int,r::Int,s::Int,theta::Float64,d;kwargs...)
    @unpack FoI = kwargs;
    seed = 123;
    uncertParam = "1";
    return ControlVariate(obj,N_samples_s,N_samples_diff,seed,uncertParam,r,s,theta,d;FoI);
end

function check_level_setup_DLR(obj::UQSetup, ℓ::Int)
    # Check if the level ℓ is already set up in Problem_levels
    problem = obj.Problem_levels["0"]["DLRASetup"].solver.settings.problem;
    if haskey(obj.Problem_levels, "$ℓ")
        
    else
        if haskey(obj.Problem_levels, "$(ℓ-1)")
            if problem == "Planesource"  || problem == "PlanePulse" || problem == "GaussianPulse"  
                Nx = (obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.Nx-1)*2 + 1;
                nPN = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.nPN
                s = settings(Nx, nPN);
                s.r = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.r;
            
                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.uncertParam;
                
                DLRASetup = DLRAIntegratorSetup(Solver,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].setupIC,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].K_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].L_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].S_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].pre_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].post_step);
          
                DLRASetup.r = max(round(min(DLRASetup.solver.settings.Nx,DLRASetup.solver.settings.nPN)/2),2);
                DLRASetup.ϑ = s.dx^2;
                DLRASetup.cη = 5.0;

                obj.Problem_levels["$ℓ"] = Dict(
                    "DLRASetup" => DLRASetup,
                     "rank_diff" => 0,
                     "rank_sum"  => 0
                    );
                Solver = nothing;
                s = nothing;
                DLRASetup = nothing;
                    # println("Setting up level $ℓ with r = ", settings.dx)
            elseif problem == "Linesource" || problem == "Lattice"    
                Nx = (obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.Nx-2)*2 + 2;
                Ny = (obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.Ny-2)*2 + 2;
              
                s = settings(Nx,Ny, 1000,problem);
              
                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.uncertParam;
              
                DLRASetup = DLRAIntegratorSetup(Solver,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].setupIC,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].K_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].L_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].S_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].pre_step,obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].post_step);
                DLRASetup.r = max(round(min(DLRASetup.solver.settings.Nx,DLRASetup.solver.pn.nTotalEntries)/2),2); # Try to make this more general
                DLRASetup.ϑ = min(s.dx^2,s.dy^2);
                DLRASetup.cη = 5.0;

                obj.Problem_levels["$ℓ"] = Dict(
                     "DLRASetup" => DLRASetup,
                     "rank_diff" => 0,
                     "rank_sum"  => 0
                    );
            elseif problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                Nx = (obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.Nx-2)*2 + 2;
              
                s = settings(Nx, problem = problem);
                s.r = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.r;
                s.ϑ = s.dx^2;
                
                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.uncertParam;
                
                obj.Problem_levels["$ℓ"] = Dict(
                    "solver" => Solver,
                     "rank_diff" => 0
                    );
            elseif problem == "LinearAdvection1d"
                Nx = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.Nx*2;
                c = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.c;
                T = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.T;
                CFL = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.CFL;
                obj.Problem_levels["$ℓ"] = Dict(
                    "solver" => LinearAdvection1d(Nx, -2.0, 2.0, c, T,CFL);
                    );
            elseif problem == "Hohlraum"
                prev_solver = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver
                prev_model  = prev_solver.model
                reset_geom!(prev_model)   # ensure nominal geometry before creating finer level
                Nx_ℓ = prev_solver.settings.Nx * 2  # double the resolution
                Nx_ℓ = 13 * max(1, round(Int, Nx_ℓ / 13))  # snap to multiple of 13
                new_model  = HohlraumModel(prev_model.geom, Nx_ℓ, Nx_ℓ;
                                           tEnd = prev_solver.tEnd,
                                           cfl  = prev_model.cfl)
                new_solver = Main.Solver(new_model; Nv = prev_solver.sn.Nv)
                new_setup  = DLRAIntegratorSetup(new_solver,
                                                 obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].setupIC;
                                                 computeQoI = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].computeQoI)
                new_setup.r   = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].r
                new_setup.ϑ   = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].ϑ
                new_setup.cη  = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].cη
                new_setup.κ   = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].κ
                obj.Problem_levels["$ℓ"] = Dict(
                    "DLRASetup" => new_setup,
                    "rank_diff" => 0,
                    "rank_sum"  => 0,
                )
            else
                println("Problem not coded yet")
            end
        else
            println("Problem level $(ℓ-1) not set up yet")
        end
    end
    return nothing;
end

function check_level_setup(obj::UQSetup, ℓ::Int)
    # Check if the level ℓ is already set up in Problem_levels
    problem = obj.Problem_levels["0"]["solver"].settings.problem;
    if haskey(obj.Problem_levels, "$ℓ")
        
    else
        if haskey(obj.Problem_levels, "$(ℓ-1)")
            if problem == "Planesource" || problem == "PlanePulse" || problem == "GaussianPulse"
                Nx = (obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx-1)*2 + 1;
                nPN = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.nPN

                s = settings(Nx, nPN);
                s.r = max(round(min(s.dx,s.nPN)/2),2);
                s.epsAdapt = s.dx^2; # The tolerance for the rank adaptive integrator

                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["solver"].uncertParam;
                
                obj.Problem_levels["$ℓ"] = Dict(
                    "solver" => Solver,
                     "rank_diff" => 0,
                     "rank_sum"  => 0
                    );
                    # println("Setting up level $ℓ with r = ", settings.dx)
            elseif problem == "Linesource" || problem == "Lattice"
                Nx = (obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx-2)*2 + 2;
                Ny = (obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Ny-2)*2 + 2;

                s = settings(Nx,Ny, 1000,obj.Problem_levels["$(ℓ-1)"]["solver"].settings.problem);
                s.r = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.r;

                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["solver"].uncertParam;
                Solver.settings.epsAdapt = (min(s.dx,s.dy))^2;
            
                obj.Problem_levels["$ℓ"] = Dict(
                     "solver" => Solver,
                     "rank_diff" => 0
                    );
            elseif problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                Nx = (obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx-2)*2 + 2;

                s = settings(Nx, problem = problem);
                s.r = max(round(min(s.dx,s.N)/2),2);
                s.ϑ = s.dx^2;
                s.rMax = floor(Int, (s.N - 2) / 2);  # ensure 2r ≤ N-2 in BUG step

                Solver = solver(s);
                Solver.uncertParam = obj.Problem_levels["$(ℓ-1)"]["solver"].uncertParam;
                
                obj.Problem_levels["$ℓ"] = Dict(
                    "solver" => Solver,
                     "rank_diff" => 0
                    );
            elseif problem == "LinearAdvection1d"
                Nx = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx*2;
                c = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.c;
                T = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.T;
                CFL = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.CFL;
                obj.Problem_levels["$ℓ"] = Dict(
                    "solver" => LinearAdvection1d(Nx, -2.0, 2.0, c, T,CFL);
                    );
            else
                println("Problem not coded yet")
            end
        else
            println("Problem level $(ℓ-1) not set up yet")
        end
    end
    return nothing;
end

function ComputeDiff_DLR(Qf_ℓ0::Array{Float64,1},dQ::Array{Float64,1},Problem_levels::Dict,ℓ::Int,Interpolate::String,j::Int)
    # If the FoI is an array, the QoI is the difference between the two levels
    # with the coarse level being linearly interpolated onto the fine level.
    if ℓ == 0
        # If the level is 0, return the QoI as the fine level FoI
        return dQ;
    elseif length(dQ) == 1
        # Scalar QoI (e.g. total absorption): no spatial interpolation needed
        return dQ .- Qf_ℓ0;
    else
        # If the level is not 0, interpolate the coarse level onto the fine level
        if Interpolate == "1D"
            if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                x_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.xMid;
                x_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.xMid;
                Qf_ℓ0_interp = linear_interpolation(x_coarse, Qf_ℓ0, extrapolation_bc = Line());
                return dQ .- Qf_ℓ0_interp(x_fine);
            else
                x_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.x;
                x_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.x;
                Qf_ℓ0_interp = linear_interpolation(x_coarse, Qf_ℓ0, extrapolation_bc = Line());
                return dQ .- Qf_ℓ0_interp(x_fine);
            end
        elseif Interpolate == "2D"
            x_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.xMid;
            y_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.yMid;

            x_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.xMid;
            y_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.yMid;
            nx = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.NCellsX;
            ny = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.NCellsY;

            coarseGrid = RectangleGrid(x_coarse,y_coarse);
            fineGrid = RectangleGrid(x_fine,y_fine);

            y = zeros(nx,ny);
            
            for i = 1:nx
                for j = 1:ny
                    z = fineGrid[i,j];
                    y[i,j] = GridInterpolations.interpolate(coarseGrid,Qf_ℓ0,z);
                end
            end       
            return dQ .- vec(y);

        elseif Interpolate == "3D"
            x_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.xMid;
            y_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.yMid;
            z_coarse = Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.settings.zMid;

            x_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.xMid;
            y_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.yMid;
            z_fine = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.zMid;

            nx = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.NCellsX;
            ny = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.NCellsY;
            nz = Problem_levels["$(ℓ)"]["DLRASetup"].solver.settings.NCellsZ;

            coarseGrid = RectangleGrid(x_coarse,y_coarse,z_coarse);
            fineGrid = RectangleGrid(x_fine,y_fine,z_fine);
            y = zeros(length(dQ));
            
            for i = 1:nx
                for j = 1:ny
                    for k = 1:nz
                        z = fineGrid[i,j,k];
                        grid_val = GridInterpolations.interpolate(coarseGrid,Qf_ℓ0,z);
                        y[vectorIndex(nx,ny,i,j,k)] = grid_val;
                    end
                end
            end            
            return dQ .- y;
        end
    end
end

function ComputeDiff(Qf_ℓ0::Array{Float64,1},dQ::Array{Float64,1},Problem_levels::Dict,ℓ::Int,Interpolate::String,j::Int)
    # If the FoI is an array, the QoI is the difference between the two levels
    # with the coarse level being linearly interpolated onto the fine level.
    problem = Problem_levels["0"]["solver"].settings.problem;
    if ℓ == 0
        # If the level is 0, return the QoI as the fine level FoI
        return dQ;
    elseif length(dQ) == 1
        # Scalar QoI: no spatial interpolation needed
        return dQ .- Qf_ℓ0;
    else
        # If the level is not 0, interpolate the coarse level onto the fine level
        if Interpolate == "1D"
            if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                x_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.xMid;
                x_fine = Problem_levels["$(ℓ)"]["solver"].settings.xMid;
                Qf_ℓ0_interp = linear_interpolation(x_coarse, Qf_ℓ0, extrapolation_bc = Line());
                return dQ .- Qf_ℓ0_interp(x_fine);
            else
                x_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.x;
                x_fine = Problem_levels["$(ℓ)"]["solver"].settings.x;
                Qf_ℓ0_interp = linear_interpolation(x_coarse, Qf_ℓ0, extrapolation_bc = Line());
                return dQ .- Qf_ℓ0_interp(x_fine);
            end
        elseif Interpolate == "2D"
            x_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.xMid;
            y_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.yMid;

            x_fine = Problem_levels["$(ℓ)"]["solver"].settings.xMid;
            y_fine = Problem_levels["$(ℓ)"]["solver"].settings.yMid;
            nx = Problem_levels["$(ℓ)"]["solver"].settings.NCellsX;
            ny = Problem_levels["$(ℓ)"]["solver"].settings.NCellsY;

            coarseGrid = RectangleGrid(x_coarse,y_coarse);
            fineGrid = RectangleGrid(x_fine,y_fine);

            y = zeros(nx,ny);
            
            for i = 1:nx
                for j = 1:ny
                    z = fineGrid[i,j];
                    y[i,j] = GridInterpolations.interpolate(coarseGrid,Qf_ℓ0,z);
                end
            end       
            return dQ .- vec(y);

        elseif Interpolate == "3D"
            x_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.xMid;
            y_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.yMid;
            z_coarse = Problem_levels["$(ℓ-1)"]["solver"].settings.zMid;

            x_fine = Problem_levels["$(ℓ)"]["solver"].settings.xMid;
            y_fine = Problem_levels["$(ℓ)"]["solver"].settings.yMid;
            z_fine = Problem_levels["$(ℓ)"]["solver"].settings.zMid;

            nx = Problem_levels["$(ℓ)"]["solver"].settings.NCellsX;
            ny = Problem_levels["$(ℓ)"]["solver"].settings.NCellsY;
            nz = Problem_levels["$(ℓ)"]["solver"].settings.NCellsZ;

            coarseGrid = RectangleGrid(x_coarse,y_coarse,z_coarse);
            fineGrid = RectangleGrid(x_fine,y_fine,z_fine);
            y = zeros(length(dQ));
            
            for i = 1:nx
                for j = 1:ny
                    for k = 1:nz
                        z = fineGrid[i,j,k];
                        grid_val = GridInterpolations.interpolate(coarseGrid,Qf_ℓ0,z);
                        y[vectorIndex(nx,ny,i,j,k)] = grid_val;
                    end
                end
            end            
            return dQ .- y;
        end
    end
end


function ComputeDiff(Qf_ℓ0::CuArray{Float32,1},dQ::CuArray{Float32,1},Problem_levels::Dict,ℓ::Int)
    # If the FoI is an array, the QoI is the difference between the two levels 
    # with the coarse level being linearly interpolated onto the fine level.
    if ℓ == 0
        # If the level is 0, return the QoI as the fine level FoI
        return dQ;
    else 
        # If the level is not 0, interpolate the coarse level onto the fine level
        x_coarse = Float32.(Problem_levels["$(ℓ-1)"]["solver"].settings.x);
        x_fine = Float32.(Problem_levels["$(ℓ)"]["solver"].settings.x);
        Qf_ℓ0_interp = linear_interpolation(x_coarse, Qf_ℓ0);
        return dQ .- cu(Qf_ℓ0_interp(x_fine));
    end
end

function ComputeDiff(Qf_ℓ0::Float64,dQ::Float64,Problem_levels::Dict,ℓ::Int)
    # If the FoI is a scalar, the QoI is simply the difference between the two levels.
    if ℓ == 0
        return dQ;
    else 
        return dQ - Qf_ℓ0;
    end
end

function ComputeSample_DLR(obj,ℓ::Int,sample::Array{Float64,1},j)
    Nx_ℓ1 = Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize));
    Qf_ℓ1 = zeros(Float64,Nx_ℓ1); # Dont pre-allocate memory

    DLRASetup_ℓ1 = obj.Problem_levels["$ℓ"]["DLRASetup"];
    DLRASetup_ℓ1.solver.sample = sample;
    f_ℓ1,rVec_ℓ1 = obj.Numerical_solver(DLRASetup_ℓ1,sample);  #[Int((length(obj.x)+1)/2)];

    Qf_ℓ1s = [Sol_to_FoI(f_ℓ1, foi) for foi in obj.FoI];
    dQs    = [copy(q) for q in Qf_ℓ1s];

    rVec_ℓ0 = zeros(size(rVec_ℓ1));
    if ℓ != 0
        Nx_ℓ0 = Int(prod(obj.Problem_levels["$(ℓ-1)"]["DLRASetup"].solver.gridSize));
        DLRASetup_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["DLRASetup"];
        DLRASetup_ℓ0.solver.sample = sample;
        f_ℓ0,rVec_ℓ0 = obj.Numerical_solver(DLRASetup_ℓ0,sample);

        for (k, foi) in enumerate(obj.FoI)
            Qf_ℓ0 = Sol_to_FoI(f_ℓ0, foi);
            if obj.λ == 0
                dQs[k] = ComputeDiff_DLR(Qf_ℓ0, dQs[k], obj.Problem_levels, ℓ, obj.Interpolate, j);
            else
                dQs[k] = ComputeDiff_DLR(obj.λ .* Qf_ℓ0, dQs[k], obj.Problem_levels, ℓ, obj.Interpolate, j);
            end
        end
    end
    r_Qf_ℓ1 = maximum(rVec_ℓ1[2,:]);
    obj.Problem_levels["$ℓ"]["rank_diff"] = max(obj.Problem_levels["$ℓ"]["rank_diff"],r_Qf_ℓ1);
    obj.Problem_levels["$ℓ"]["rank_sum"] += r_Qf_ℓ1;
    return dQs, Qf_ℓ1s;
end

function ComputeSample(obj,ℓ::Int,sample::Array{Float64,1},j)
    solver_ℓ1 = obj.Problem_levels["$ℓ"]["solver"];
    solver_ℓ1.sample = sample;

    f_ℓ1,rVec_ℓ1 = obj.Numerical_solver(solver_ℓ1,sample);

    Qf_ℓ1s = [Sol_to_FoI(f_ℓ1, foi) for foi in obj.FoI];
    dQs    = [copy(q) for q in Qf_ℓ1s];

    if ℓ != 0
        solver_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"];
        solver_ℓ0.sample = sample;
        f_ℓ0,rVec_ℓ0 = obj.Numerical_solver(solver_ℓ0,sample);

        for (k, foi) in enumerate(obj.FoI)
            Qf_ℓ0 = Sol_to_FoI(f_ℓ0, foi);
            if obj.λ == 0
                dQs[k] = ComputeDiff(Qf_ℓ0, dQs[k], obj.Problem_levels, ℓ, obj.Interpolate, j);
            else
                dQs[k] = ComputeDiff(obj.λ .* Qf_ℓ0, dQs[k], obj.Problem_levels, ℓ, obj.Interpolate, j);
            end
        end
    end
    r_Qf_ℓ1 = maximum(rVec_ℓ1[2,end]);
    obj.Problem_levels["$ℓ"]["rank_diff"] = max(obj.Problem_levels["$ℓ"]["rank_diff"],r_Qf_ℓ1);
    return dQs, Qf_ℓ1s;
end

function CUComputeSample(obj,ℓ::Int,sample::Array{Float64,1})

    if obj.Numerical_solver == solveAugBUG || obj.Numerical_solver == CUsolveAugBUG
        Nx_ℓ1 = obj.Problem_levels["$ℓ"]["solver"].settings.Nx;
        Qf_ℓ1 = cu(zeros(Float64,Nx_ℓ1));
        solver_ℓ1 = obj.Problem_levels["$ℓ"]["solver"];
        f_ℓ1,rVec_ℓ1 = obj.Numerical_solver(solver_ℓ1,sample);  #[Int((length(obj.x)+1)/2)];
        Qf_ℓ1 .= Sol_to_FoI(f_ℓ1,obj.FoI);
        dQ = Qf_ℓ1;
        rVec_ℓ0 = zeros(size(rVec_ℓ1));

        if ℓ != 0
            Nx_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx;
            solver_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"];
            f_ℓ0,rVec_ℓ0 = obj.Numerical_solver(solver_ℓ0,sample);  #[Int((length(obj.x)+1)/2)];
            Qf_ℓ0 = cu(Sol_to_FoI(f_ℓ0,obj.FoI));
            dQ = ComputeDiff(Qf_ℓ0, dQ, obj.Problem_levels, ℓ,obj.Interpolate);
        end

        r_Qf_ℓ1 = rVec_ℓ1[2,end] - rVec_ℓ0[2,end];
        obj.Problem_levels["$ℓ"]["rank_diff"] += r_Qf_ℓ1;

    else
        Nx_ℓ1 = obj.Problem_levels["$ℓ"]["solver"].settings.Nx;
        Qf_ℓ1 = CUDA.zeros(Float64,Nx_ℓ1);
        solver_ℓ1 = obj.Problem_levels["$ℓ"]["solver"];
        Qf_ℓ1 .= Sol_to_FoI(obj.Numerical_solver(solver_ℓ1,sample),obj.FoI) #[Int((length(obj.x)+1)/2)];
        Qf_ℓ1 = cu(Qf_ℓ1);
        dQ = Qf_ℓ1;
        
        if ℓ != 0
            Nx_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"].settings.Nx;
            solver_ℓ0 = obj.Problem_levels["$(ℓ-1)"]["solver"];
            Qf_ℓ0 = cu(Sol_to_FoI(obj.Numerical_solver(solver_ℓ0,sample),obj.FoI));
            dQ = ComputeDiff(Qf_ℓ0, dQ, Problem_levels, ℓ,obj.Interpolate);
        end
    end
    return dQ,Qf_ℓ1;
end

function ComputeMean_DLR(obj,Dict_levels)
    L = 0;
    for key in keys(Dict_levels)
        try
            key_int = parse(Int,key);
            if key_int >= L
                L = key_int;
            end
        catch   
        end
    end

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
        mean_ℓ_raw = Dict_levels["$ℓ"]["mean"];
        mean_ℓ_vec = isa(mean_ℓ_raw, Vector{Vector{Float64}}) ? mean_ℓ_raw[1] : mean_ℓ_raw;
        mean_ℓ_interp = linear_interpolation(x_ℓ, mean_ℓ_vec, extrapolation_bc = Line());
        mean_L .+= mean_ℓ_interp(x_L);
    end
    return mean_L
    elseif obj.Interpolate == "2D"
    mean_L_raw = Dict_levels["$L"]["mean"];
    mean_L_vec = isa(mean_L_raw, Vector{Vector{Float64}}) ? mean_L_raw[1] : mean_L_raw;
    mean_L = zeros(length(mean_L_vec));
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
        mean_ℓ_raw = Dict_levels["$ℓ"]["mean"];
        mean_ℓ = isa(mean_ℓ_raw, Vector{Vector{Float64}}) ? mean_ℓ_raw[1] : mean_ℓ_raw;
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
    return mean_L;
    else
    Throw(ArgumentError("Plotting has not be coded yet"))
    end
end

function ComputeMean(obj,Dict_levels)
    L = 0;
    for key in keys(Dict_levels)
        try
            key_int = parse(Int,key);
            if key_int >= L
                L = key_int;
            end
        catch   
        end
    end

    if obj.Interpolate == "1D"
        settings_L = obj.Problem_levels["$L"]["solver"].settings;
        if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
            x_L = settings_L.xMid;
        else
            x_L = settings_L.x;
        end
        mean = zeros(Float64, length(x_L));
        
        for ℓ = 0:L
            settings_ℓ = obj.Problem_levels["$ℓ"]["solver"].settings;
            if problem == "shock" || problem == "sqrt" || problem == "KowalskiTorrihon"
                x_ℓ = settings_ℓ.xMid;
            else
                x_ℓ = settings_ℓ.x;
            end
            mean_ℓ_interp = linear_interpolation(x_ℓ, Dict_levels["$ℓ"]["mean"], extrapolation_bc = Line());
            mean .+= mean_ℓ_interp(x_L);
        end
        return mean;
    elseif obj.Interpolate == "2D"
        mean = zeros(length(Dict_levels["$L"]["mean"]));
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
            mean_ℓ = Dict_levels["$ℓ"]["mean"];
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
        return mean;
    else
        Throw(ArgumentError("Plotting has not be coded yet"))
    end
end

function MLMC_fixedLevels(obj::UQSetup)
    # Estimating variances, costs, and sample sizes for the two-level Monte Carlo method    
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;

    println("Provide the maximum levels for the MLMC method (L):");
    L = readline(); # Number of levels
    if L == ""
        L = 5; # Default value
    elseif L == "k"
        println("Exiting the MLMC_fixedLevels method without computations.")
        return;
    else
        L = parse(Int, L);
    end
    
    println("Number of levels: ", L)

    for ℓ = 0:L
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            check_level_setup_DLR(obj, ℓ);
        else
            check_level_setup(obj, ℓ);
        end
    end
    
    if haskey(obj.Problem_levels["0"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    end
    α, β = 0.0, 0.0;

    time_start = time();
    while sum([Dict_levels["$ℓ"]["Add_samples"] for ℓ = 0:L]) > 0
        for ℓ = 0:L
            dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            if haskey(obj.Problem_levels["0"],"DLRASetup")
                dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
            else
                dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
            end
            
            count = Dict_levels["$ℓ"]["N_samples"];
            mean_ℓ =  Dict_levels["$ℓ"]["mean"];
            delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
            var_ℓ = Dict_levels["$ℓ"]["var"];
            
            time = 0.0;
            time_list = [];
            if dN_ℓ >0
                for n = 1:dN_ℓ
                    count += 1;
                    sample = [rand(pdf[i]) for i = 1:length(pdf)];

                    t = @elapsed begin
                        if haskey(obj.Problem_levels["0"],"DLRASetup")
                            dQ,Qf = ComputeSample_DLR(obj,ℓ,sample,n);
                        else
                            dQ,Qf = ComputeSample(obj,ℓ,sample,n);
                        end
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
                    # track E[Q_ℓ] (absolute fine-level solution) at every level
                    delta_Qf .= (Qf - mean_Qf_ℓ);
                    mean_Qf_ℓ .+= delta_Qf./count;
                end
                N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
                Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
                var_ℓ = var_ℓ*dx/(N_ℓ - 1);


                Dict_levels["$ℓ"]["mean"]    = mean_ℓ;
                Dict_levels["$ℓ"]["mean_Qf"] = mean_Qf_ℓ;
                Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
                Dict_levels["$ℓ"]["cost"] = minimum(time_list);
                Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
                if obj.Verbose
                    println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
                    println("------------------------------------------------------------------")
                end
            end
            
            if obj.Plot[2]
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    plot_mean_DLR(obj,Dict_levels,ℓ);
                    plot_dQ_DLR(obj,Dict_levels,ℓ);
                else
                    plot_mean(obj,Dict_levels,ℓ);
                    plot_dQ(obj,Dict_levels,ℓ);
                end
            end
        end
    

        for ℓ = 2:L
            Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
            Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
        end

        if obj.Verbose
            println("Estimating the rates α, β, and γ from the variances and costs")
        end
        α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
        β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
        lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
        γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
        cγ = 2^(LinearRegression.bias(lr_γ)[1])
        if obj.Verbose
            println("α: ", α)
            println("β: ", β)
            println("γ: ", γ)
            println("cγ: ", cγ)
            println("------------------------------------------------------------------")

            println("Estimating the optimal number of samples for each level")
            println("------------------------------------------------------------------")
        end

        Nsamples_new = zeros(Int, L);
        for ℓ = 0:L
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
            if Dict_levels["$ℓ"]["Add_samples"] > 0
                if obj.Verbose
                    println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                end
            end
        end 
        Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
    end
    
    time_end = time();
    total_var = 0.0;
    for ℓ = 0:L
        total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
    end

    if obj.Verbose
        println("Total time taken: ", (time_end - time_start), " seconds")
    end

    total_cost = 0.0;
    for ℓ = 0:L
        total_cost += Dict_levels["$ℓ"]["cost"]*Dict_levels["$ℓ"]["N_samples"];
    end

    if haskey(obj.Problem_levels["0"],"DLRASetup")
        settings_L = obj.Problem_levels["$L"]["DLRASetup"].solver.settings;
    else
        settings_L = obj.Problem_levels["$L"]["solver"].settings;
    end

    mean = zeros(Float64, settings_L.Nx);
    for ℓ = 0:L
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            settings_ℓ = obj.Problem_levels["$ℓ"]["DLRASetup"].solver.settings;
        else
            settings_ℓ = obj.Problem_levels["$ℓ"]["solver"].settings;
        end
        mean_ℓ_interp = linear_interpolation(settings_ℓ.x, Dict_levels["$ℓ"]["mean"], extrapolation_bc = Line());
        mean .+= mean_ℓ_interp(settings_L.x);
    end
    
    if obj.Plot[1] 
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            plot_mean_DLR(obj,Dict_levels,L);
        else
            plot_mean(obj,Dict_levels,L);
        end
    end

    return Dict_levels;
end

function MLMC_adaptive_step(obj,Dict_levels,ℓ)
    dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
    Nx = length(Dict_levels["$ℓ"]["mean"]);
    dx = obj.Problem_levels["$ℓ"]["solver"].settings.dx;
    pdf = obj.pdf;
    
    count = Dict_levels["$ℓ"]["N_samples"];
    mean_ℓ =  Dict_levels["$ℓ"]["mean"];
    delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
    var_ℓ = Dict_levels["$ℓ"]["var"];

    time = 0.0;
    if dN_ℓ >0
        
        Threads.@threads for n = 1:dN_ℓ
            count += 1;
            sample = [rand(pdf[i]) for i = 1:length(pdf)];

            t = @elapsed begin
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
                else
                    dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
                end
            end
            dQ = dQs[1]; Qf = Qfs[1];

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
        Dict_levels["$ℓ"]["cost"] = time/N_ℓ;
        Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
        if obj.Verbose
            println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", time/N_ℓ)
            println("------------------------------------------------------------------")
        end
    end
    return Dict_levels;
end

function MLMC_adaptive(obj::UQSetup)
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    L = 2; # Number of levels
    for ℓ = 0:L
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            check_level_setup_DLR(obj, ℓ);
        else
            check_level_setup(obj, ℓ);
        end
    end
    if haskey(obj.Problem_levels["0"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => [zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))) for _ in obj.FoI], "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => [zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))) for _ in obj.FoI], "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    end
    α, β = 0.0, 0.0;

    time_start = time();
    while sum([Dict_levels["$ℓ"]["Add_samples"] for ℓ = 0:L]) > 0
        # global α, β
        for ℓ = 0:L
            dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            if haskey(obj.Problem_levels["0"],"DLRASetup")
                dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
            else
                dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
            end
            
            count    = Dict_levels["$ℓ"]["N_samples"];
            means_ℓ  = Dict_levels["$ℓ"]["mean"];          # Vector{Vector{Float64}}, one per FoI
            deltas_ℓ = [zeros(Float64, length(m)) for m in means_ℓ];
            vars_ℓ   = zeros(Float64, length(obj.FoI));

            time = 0.0;
            time_list = [];
            if dN_ℓ >0
                for n = 1:dN_ℓ
                    count += 1;
                    sample = [rand(pdf[i]) for i = 1:length(pdf)];

                    t = @elapsed begin
                        if haskey(obj.Problem_levels["0"],"DLRASetup")
                            dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
                        else
                            dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
                        end
                    end
                    time_list = push!(time_list, t);
                    if n > 6
                        time += t;
                    end

                    for k = 1:length(obj.FoI)
                        q = ℓ == 0 ? Qfs[k] : dQs[k];
                        deltas_ℓ[k] .= q .- means_ℓ[k];
                        means_ℓ[k] .+= deltas_ℓ[k] ./ count;
                        vars_ℓ[k]  += deltas_ℓ[k]' * (q .- means_ℓ[k]);
                    end
                end
                N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
                Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
                var_ℓ = maximum(vars_ℓ) * dx / (N_ℓ - 1);


                Dict_levels["$ℓ"]["mean"] = means_ℓ;
                Dict_levels["$ℓ"]["var"] = max(0, var_ℓ);
                Dict_levels["$ℓ"]["cost"] = minimum(time_list);
                Dict_levels["$ℓ"]["bias"] = maximum(sqrt(m' * m * dx) for m in means_ℓ);
                if obj.Verbose
                    println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
                    println("------------------------------------------------------------------")
                end
            end
            if obj.Plot[2]
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    plot_mean_DLR(obj,Dict_levels,ℓ);
                    plot_dQ_DLR(obj,Dict_levels,ℓ);
                else
                    plot_mean(obj,Dict_levels,ℓ);
                    plot_dQ(obj,Dict_levels,ℓ);
                end
            end
        end

        for ℓ = 2:L
            Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
            Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
        end
        if obj.Verbose
            println("Estimating the rates α, β, and γ from the variances and costs")
        end
        α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
        β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
        lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
        γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
        cγ = 2^(LinearRegression.bias(lr_γ)[1])
        if obj.Verbose
            println("α: ", α)
            println("β: ", β)
            println("γ: ", γ)
            println("cγ: ", cγ)
            println("------------------------------------------------------------------")

        
            println("Estimating the optimal number of samples for each level")
            println("------------------------------------------------------------------")
        end
        for ℓ = 0:L
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
            if Dict_levels["$ℓ"]["Add_samples"] > 0
                if obj.Verbose
                    println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                end
            end
        end 

        if sum([Dict_levels["$ℓ"]["Add_samples"]>0.01*Dict_levels["$ℓ"]["N_samples"] for ℓ = 0:L]) == 0 
            rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
            if rem > ε/sqrt(2)
                if obj.Verbose
                    println("Adding a new level")
                end
                L = L + 1;
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    check_level_setup_DLR(obj, L);
                else
                    check_level_setup(obj, L);
                end
                obj.Problem_levels["$L"]["rank_diff"] = 0;
                obj.Problem_levels["$L"]["rank_sum"]  = 0;
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    n_L = Int(prod(obj.Problem_levels["$L"]["DLRASetup"].solver.gridSize));
                    Dict_levels["$L"] = Dict("mean" => [zeros(Float64,n_L) for _ in obj.FoI], "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L), "N_samples" => 0, "Add_samples" => 0);
                else
                    n_L = Int(prod(obj.Problem_levels["$L"]["solver"].gridSize));
                    Dict_levels["$L"] = Dict("mean" => [zeros(Float64,n_L) for _ in obj.FoI], "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L), "N_samples" => 0, "Add_samples" => 0);
                end
                for ℓ = 0:L
                    Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]+1);
                    if Dict_levels["$ℓ"]["Add_samples"] > 0
                        if obj.Verbose
                            println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                        end
                    end
                end 
            else
                if obj.Verbose
                    println("Converged")
                end
                Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                time_end = time();
                if obj.Verbose
                    println("Total time taken: ", (time_end - time_start), " seconds")
                end
                total_var = 0.0;
                for ℓ = 0:L
                    total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                end
                if obj.Verbose
                    println("Bias: ", rem)
                    println("Total variance: ", total_var)
                end
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    settings_L = obj.Problem_levels["$L"]["DLRASetup"].solver.settings;
                else
                    settings_L = obj.Problem_levels["$L"]["solver"].settings;
                end

                if obj.Plot[1]
                    if haskey(obj.Problem_levels["0"],"DLRASetup")
                        plot_mean_DLR(obj,Dict_levels,L);
                    else
                        plot_mean(obj,Dict_levels,L);
                    end
                end
            end
        end
    end
    return Dict_levels;
end


function CUMLMC_adaptive(obj::UQSetup)
    ε = obj.ε;
    N_warm_up_samples = obj.N_samples;
    pdf = obj.pdf;
    L = 2; # Number of levels
    for ℓ = 0:L
        if haskey(obj.Problem_levels["0"],"DLRASetup")
            check_level_setup_DLR(obj, ℓ);
        else
            check_level_setup(obj, ℓ);
        end
    end
    if haskey(obj.Problem_levels["0"],"DLRASetup")
        Dict_levels = Dict("$ℓ" => Dict("mean" => [zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))) for _ in obj.FoI], "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    else
        Dict_levels = Dict("$ℓ" => Dict("mean" => [zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["solver"].gridSize))) for _ in obj.FoI], "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
    end
    α, β = 0.0, 0.0;

    time_start = time();
    while sum([Dict_levels["$ℓ"]["Add_samples"] for ℓ = 0:L]) > 0
        # global α, β
        for ℓ = 0:L
            dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
            if haskey(obj.Problem_levels["0"],"DLRASetup")
                dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
            else
                dx = prod(obj.Problem_levels["$ℓ"]["solver"].gridWidth);
            end

            count    = Dict_levels["$ℓ"]["N_samples"];
            means_ℓ  = Dict_levels["$ℓ"]["mean"];          # Vector{Vector{Float64}}, one per FoI
            deltas_ℓ = [zeros(Float64, length(m)) for m in means_ℓ];
            vars_ℓ   = zeros(Float64, length(obj.FoI));

            time = 0.0;
            time_list = [];
            if dN_ℓ >0
                for n = 1:dN_ℓ
                    count += 1;
                    sample = [rand(pdf[i]) for i = 1:length(pdf)];

                    t = @elapsed begin
                        if haskey(obj.Problem_levels["0"],"DLRASetup")
                            dQs,Qfs = ComputeSample_DLR(obj,ℓ,sample,n);
                        else
                            dQs,Qfs = ComputeSample(obj,ℓ,sample,n);
                        end
                    end
                    time_list = push!(time_list, t);
                    if n > 6
                        time += t;
                    end

                    for k = 1:length(obj.FoI)
                        q = ℓ == 0 ? Qfs[k] : dQs[k];
                        deltas_ℓ[k] .= q .- means_ℓ[k];
                        means_ℓ[k] .+= deltas_ℓ[k] ./ count;
                        vars_ℓ[k]  += deltas_ℓ[k]' * (q .- means_ℓ[k]);
                    end
                end
                N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
                Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
                var_ℓ = maximum(vars_ℓ) * dx / (N_ℓ - 1);


                Dict_levels["$ℓ"]["mean"] = means_ℓ;
                Dict_levels["$ℓ"]["var"] = max(0, var_ℓ);
                Dict_levels["$ℓ"]["cost"] = minimum(time_list);
                Dict_levels["$ℓ"]["bias"] = maximum(sqrt(m' * m * dx) for m in means_ℓ);
                if obj.Verbose
                    println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
                    println("------------------------------------------------------------------")
                end
            end
            if obj.Plot[2]
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    plot_mean_DLR(obj,Dict_levels,ℓ);
                    plot_dQ_DLR(obj,Dict_levels,ℓ);
                else
                    plot_mean(obj,Dict_levels,ℓ);
                    plot_dQ(obj,Dict_levels,ℓ);
                end
            end
        end

        for ℓ = 2:L
            Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
            Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
        end
        if obj.Verbose
            println("Estimating the rates α, β, and γ from the variances and costs")
        end
        α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
        β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
        lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
        γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
        cγ = 2^(LinearRegression.bias(lr_γ)[1])
        if obj.Verbose
            println("α: ", α)
            println("β: ", β)
            println("γ: ", γ)
            println("cγ: ", cγ)
            println("------------------------------------------------------------------")


            println("Estimating the optimal number of samples for each level")
            println("------------------------------------------------------------------")
        end
        for ℓ = 0:L
            Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
            if Dict_levels["$ℓ"]["Add_samples"] > 0
                if obj.Verbose
                    println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                end
            end
        end

        if sum([Dict_levels["$ℓ"]["Add_samples"]>0.01*Dict_levels["$ℓ"]["N_samples"] for ℓ = 0:L]) == 0
            rem = maximum([Dict_levels["$ℓ"]["bias"]*2^(α*(ℓ-L)) for ℓ = L-2:L])/(2^α - 1);
            if rem > ε/sqrt(2)
                if obj.Verbose
                    println("Adding a new level")
                end
                L = L + 1;
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    check_level_setup_DLR(obj, L);
                else
                    check_level_setup(obj, L);
                end
                obj.Problem_levels["$L"]["rank_diff"] = 0;
                obj.Problem_levels["$L"]["rank_sum"]  = 0;
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    n_L = Int(prod(obj.Problem_levels["$L"]["DLRASetup"].solver.gridSize));
                    Dict_levels["$L"] = Dict("mean" => [zeros(Float64,n_L) for _ in obj.FoI], "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L), "N_samples" => 0, "Add_samples" => 0);
                else
                    n_L = Int(prod(obj.Problem_levels["$L"]["solver"].gridSize));
                    Dict_levels["$L"] = Dict("mean" => [zeros(Float64,n_L) for _ in obj.FoI], "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L), "N_samples" => 0, "Add_samples" => 0);
                end
                for ℓ = 0:L
                    Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]+1);
                    if Dict_levels["$ℓ"]["Add_samples"] > 0
                        if obj.Verbose
                            println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
                        end
                    end
                end
            else
                if obj.Verbose
                    println("Converged")
                end
                Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
                time_end = time();
                if obj.Verbose
                    println("Total time taken: ", (time_end - time_start), " seconds")
                end
                total_var = 0.0;
                for ℓ = 0:L
                    total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
                end
                if obj.Verbose
                    println("Bias: ", rem)
                    println("Total variance: ", total_var)
                end
                if haskey(obj.Problem_levels["0"],"DLRASetup")
                    settings_L = obj.Problem_levels["$L"]["DLRASetup"].solver.settings;
                else
                    settings_L = obj.Problem_levels["$L"]["solver"].settings;
                end

                if obj.Plot[1]
                    if haskey(obj.Problem_levels["0"],"DLRASetup")
                        plot_mean_DLR(obj,Dict_levels,L);
                    else
                        plot_mean(obj,Dict_levels,L);
                    end
                end
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
    Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, obj.Problem_levels["$ℓ"]["solver"].settings.Nx), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);
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
                Dict_levels["$L"] = Dict("mean" => zeros(Float64, Nx0 * 2^(L) +1), "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L) , "N_samples" => 0, "Add_samples" => 0);
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
                if obj.Plot[1]
                    if haskey(obj.Problem_levels["0"],"DLRASetup")
                        plot_mean_DLR(obj,Dict_levels,L);
                    else
                        plot_mean(obj,Dict_levels,L);
                    end
                end
            end
        end
    end
    return Dict_levels;
end


# function Budgeted_MLMC_adaptive(obj::UQSetup)
#     ε = obj.ε;
#     ε0 = ε;
#     B = obj.B; # The total budget for the MLMC method
#     θ = obj.θ; # Splitting parameter
#     η = obj.η; # Reduction parameter
#     N_warm_up_samples = obj.N_samples;
#     pdf = obj.pdf;
#     L = 2; # Number of levels
#     for ℓ = 0:L
#         check_level_setup(obj, ℓ);
#         # obj.Problem_levels["$ℓ"]["rank_diff"] = 0; # Initialize the rank difference for each level
#     end
#     Dict_levels = Dict("$ℓ" => Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => 0.0, "cost" => 0.0, "N_samples" => 0, "Add_samples" => N_warm_up_samples) for ℓ = 0:L);

#     α, β = 0.0, 0.0;

#     time_start = time();

#     for ℓ = 0:L
#         dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
#         Nx = length(Dict_levels["$ℓ"]["mean"]);
#         dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
        
#         count = Dict_levels["$ℓ"]["N_samples"];
#         mean_ℓ =  Dict_levels["$ℓ"]["mean"];
#         delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
#         var_ℓ = Dict_levels["$ℓ"]["var"];

#         time = 0.0;
#         time_list = [];
#         if dN_ℓ >0
#             for n = 1:dN_ℓ
#                 count += 1;
#                 sample = [rand(pdf[i]) for i = 1:length(pdf)];

#                 t = @elapsed begin
#                 dQ,Qf = ComputeSample(obj,ℓ,sample);
#                 end
#                 time_list = push!(time_list, t);
#                 if n > 6
#                     time += t;
#                 end
                
#                 if ℓ == 0
#                     delta_ℓ .= (Qf - mean_ℓ);
#                     mean_ℓ .+= delta_ℓ./count;
#                     var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
#                 else
#                     delta_ℓ .= (dQ - mean_ℓ);
#                     mean_ℓ .+= delta_ℓ./count;
#                     var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
#                 end
#             end
#             N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
#             Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
#             var_ℓ = var_ℓ*dx/(N_ℓ - 1);


#             Dict_levels["$ℓ"]["mean"] = mean_ℓ;
#             Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
#             Dict_levels["$ℓ"]["cost"] = minimum(time_list);
#             Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
#             println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
#             println("------------------------------------------------------------------")
#         end
        
#     end
#     B = B - sum([Dict_levels["$ℓ"]["cost"]*Dict_levels["$ℓ"]["N_samples"] for ℓ=0:L]); 
#     ε = η * ε;
    
#     while B > 0 
#         if rem >= sqrt(1-θ)*ε
#             println("Adding a new level")
#             L = L + 1;
#             check_level_setup(obj, L);
#             obj.Problem_levels["$L"]["rank_diff"] = 0; # Initialize the rank difference for each level
#             Dict_levels["$L"] = Dict("mean" => zeros(Float64, Int(prod(obj.Problem_levels["$L"]["DLRASetup"].solver.gridSize))), "bias" => 0.0, "var" => Dict_levels["$(L-1)"]["var"]*2^(-β), "cost" => cγ*2^(γ*L) , "N_samples" => 0, "Add_samples" => 0);
#         end

#         if total_var >= θ*ε^2
#             for ℓ = 0:L
#                 Dict_levels["$ℓ"]["Add_samples"] = max(0,ceil(Int, 2/ε^2 * (sqrt(Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["cost"]) * sum([sqrt(Dict_levels["$i"]["var"]*Dict_levels["$i"]["cost"]) for i = 0:L])))-Dict_levels["$ℓ"]["N_samples"]);
#                 if Dict_levels["$ℓ"]["Add_samples"] > 0
#                     println("Additional samples on level $ℓ: ", Dict_levels["$ℓ"]["Add_samples"])
#                 end
#             end 
#         end

#         add_cost = sum([Dict_levels["$ℓ"]["Add_samples"]*Dict_levels["$ℓ"]["cost"] for ℓ=0:L]);

#         if add_cost == 0
#             ε = η * ε;
#         elseif add_cost > B
#             ε = (ε + ε0)/2;
#         else
#             for ℓ = 0:L
#                 dN_ℓ = Dict_levels["$ℓ"]["Add_samples"];
#                 Nx = length(Dict_levels["$ℓ"]["mean"]);
#                 dx = prod(obj.Problem_levels["$ℓ"]["DLRASetup"].solver.gridWidth);
                
#                 count = Dict_levels["$ℓ"]["N_samples"];
#                 mean_ℓ =  Dict_levels["$ℓ"]["mean"];
#                 delta_ℓ = zeros(Float64,length(Dict_levels["$ℓ"]["mean"]));
#                 var_ℓ = Dict_levels["$ℓ"]["var"];

#                 time = 0.0;
#                 time_list = [];
#                 if dN_ℓ > 0 
#                     for n = 1:dN_ℓ
#                         count += 1;
#                         sample = [rand(pdf[i]) for i = 1:length(pdf)];
            
#                         t = @elapsed begin
#                         dQ,Qf = ComputeSample(obj,ℓ,sample);
#                         end
#                         time_list = push!(time_list, t);
#                         if n > 6
#                             time += t;
#                         end
                        
#                         if ℓ == 0
#                             delta_ℓ .= (Qf - mean_ℓ);
#                             mean_ℓ .+= delta_ℓ./count;
#                             var_ℓ += delta_ℓ' * (Qf - mean_ℓ);
#                         else
#                             delta_ℓ .= (dQ - mean_ℓ);
#                             mean_ℓ .+= delta_ℓ./count;
#                             var_ℓ += delta_ℓ' * (dQ - mean_ℓ);
#                         end
#                     end
#                     N_ℓ = Dict_levels["$ℓ"]["N_samples"] + dN_ℓ;
#                     Dict_levels["$ℓ"]["N_samples"] = N_ℓ;
#                     var_ℓ = var_ℓ*dx/(N_ℓ - 1);
        
        
#                     Dict_levels["$ℓ"]["mean"] = mean_ℓ;
#                     Dict_levels["$ℓ"]["var"] = max(0,var_ℓ);
#                     Dict_levels["$ℓ"]["cost"] = minimum(time_list);
#                     Dict_levels["$ℓ"]["bias"] = sqrt(mean_ℓ'mean_ℓ*dx);
#                     println("Level: ", ℓ, ", Variance: ", max(0,var_ℓ),", Cost per sample: ", minimum(time_list))
#                     println("------------------------------------------------------------------")
#                 end
#             end
#             B = B - sum([Dict_levels["$ℓ"]["cost"]*Dict_levels["$ℓ"]["N_samples"] for ℓ=0:L]); 
#             for ℓ = 2:L
#                 Dict_levels["$ℓ"]["bias"] = max(Dict_levels["$ℓ"]["bias"],0.5*Dict_levels["$(ℓ-1)"]["bias"]/2^α);
#                 Dict_levels["$ℓ"]["var"] = max(Dict_levels["$ℓ"]["var"],0.5*Dict_levels["$(ℓ-1)"]["var"]/2^β);
#             end
        
#             println("Estimating the rates α, β, and γ from the variances and costs")
#             α = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["bias"] for ℓ = 1:L])))[1]);
#             β = max(0.5,-LinearRegression.slope(linregress(collect(1:L), log2.([Dict_levels["$ℓ"]["var"] for ℓ = 1:L])))[1]);
#             lr_γ = linregress(collect(0:L), log2.([Dict_levels["$ℓ"]["cost"] for ℓ = 0:L]));
#             γ = max(0.5,LinearRegression.slope(lr_γ)[1]);
#             cγ = 2^(LinearRegression.bias(lr_γ)[1])
#             println("α: ", α)
#             println("β: ", β)
#             println("γ: ", γ)
#             println("cγ: ", cγ)
#             println("------------------------------------------------------------------")
#         end
#     else
#         println("Converged")
#         Dict_levels["Convergence_rates"] = Dict("α"=> α, "β" => β, "γ" => γ, "cγ" => cγ);
#         time_end = time();
#         println("Total time taken: ", (time_end - time_start), " seconds")
#         total_var = 0.0;
#         for ℓ = 0:L
#             total_var += Dict_levels["$ℓ"]["var"]/Dict_levels["$ℓ"]["N_samples"];
#         end
#         println("Bias: ", rem)
#         println("Total variance: ", total_var)
#         solver_L = obj.Problem_levels["$L"]["DLRASetup"].solver;
#         mean = Dict_levels["$L"]["mean"];

#         plot_mean(obj,Dict_levels,L);
#     end
#     return Dict_levels;
# end