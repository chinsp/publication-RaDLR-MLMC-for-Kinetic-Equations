mutable struct DLRAIntegratorSetup
    solver

    setupIC::Function # Function to setup the initial condition

    K_step::Function # K step of the integrator
    L_step::Function # L step of the integrator
    S_step::Function # S step of the integrator

    pre_step::Function # If there are any steps that need to be done before the low-rank integrator
    post_step::Function # If there are any post DLRA steps like dose computation, moment computations etc.

    r::Int64 # Rank of the integrator
    ϑ::Float64 # The truncation tolerance for the rank-adaptive integrators
    κ::Int64 # Specifies the number of basis vectors to conserve for conservative truncation | κ = 0 implies non-conservative truncation
    cη::Float64 # Rejection tolerance

    computeQoI::Bool # Whether to compute QoI during the run (thermal radiation problems)

    # Radiation-transport constructor (K/L/S/pre/post steps supplied explicitly)
    function DLRAIntegratorSetup(solver,setupIC,K_step,L_step,S_step,pre_step,post_step)
        r = 10;
        ϑ = 0.5;
        cη = 5.0;
        κ = 0;
        new(solver,setupIC,K_step,L_step,S_step,pre_step,post_step,r,ϑ,κ,cη,false)
    end

    # Thermal-radiation constructor (GPU integrators bypass K/L/S step functions)
    function DLRAIntegratorSetup(solver, setupIC::Function; computeQoI::Bool=false)
        noop = (args...) -> nothing
        new(solver, setupIC, noop, noop, noop, noop, noop, 10, 0.5, 0, 5.0, computeQoI)
    end
end

function run(obj::DLRAIntegratorSetup,integrator_name)
    X,S,V,rVec = integrator_name(obj);
    return X,S,V,rVec;
end

function FullProblem(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    t = 0.0;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;

    g = obj.setupIC(solver);
    m,n = size(g);
    In = I(n);

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 
    rVec = zeros(Float64,3,Nt);
    for k = 1:Nt

        g = obj.K_step(solver,g,In,dt);

        rVec[1,k] = k*dt;
        rVec[2,k] = minimum(size(g));
        rVec[3,k] = sizeof(g);
    end

    return g,rVec;

end

function FullProblem(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; # Need to change this to support multiple samples
    g,rVec = FullProblem(obj);
    return g,rVec;
end


function fixedrankBUG(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    g = 0; # Free up memory

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks

    for k = 1:Nt
        # K-step 
        K = obj.K_step(solver,X*S,V,dt);
        Xhat,_ = qr(K);
        Xhat = Matrix(Xhat);
        M = Xhat'*X;

        # L-step
        Lt = obj.L_step(solver,X,S*V',dt);
        Vhat,_ = qr(Lt');
        Vhat = Matrix(Vhat);
        N = Vhat'*V;

        X,V = Xhat, Vhat;

        #S-step
        # S = obj.S_step(solver,X,S,V,dt);
        S = obj.S_step(solver,X,M*S*N',V,X,M*S*N',V,dt);

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function fixedrankBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = fixedrankBUG(obj);
    return X*S*V',rVec;
end

function fixedaugBUG(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);
    
    g = 0; # Free up memory

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks

    for k = 1:Nt
        # K-step 
        K = X*S;
        K = obj.K_step(solver,K,V,dt);
        Xhat,_ = qr([K X]);
        Xhat = Matrix(Xhat);
        M = Xhat'*X;

        # L-step
        Lt = S*V';
        Lt = obj.L_step(solver,X,Lt,dt);
        Vhat,_ = qr([Lt' V]);
        Vhat = Matrix(Vhat);
        N = Vhat'*V;

        X,V = Xhat, Vhat;

        # S-step
        # S = obj.S_step(solver,X,S,V,dt);
        S = obj.S_step(solver,X,M*S*N',V,X,M*S*N',V,dt);

        # Truncation step
        X,S,V = truncate(solver,X,S,V,ϑ,κ,true)

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function fixedaugBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = fixedaugBUG(obj);
    return X*S*V',rVec;
end

function augBUG(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    g = 0; # Free up memory

    K = zeros(size(X));
    Lt = zeros(size(V));

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks

    for k = 1:Nt
        # K-step 
        K = X*S;
        K .= obj.K_step(solver,K,V,dt);
        Xhat,_ = qr([K X]);
        Xhat = Matrix(Xhat);
        M = Xhat'*X;

        # L-step
        Lt = S*V';
        Lt .= obj.L_step(solver,X,Lt,dt);
        Vhat,_ = qr([Lt' V]);
        Vhat = Matrix(Vhat);
        N = Vhat'*V;

        X,V = Xhat, Vhat;

        # S-step
        S = obj.S_step(solver,X,M*S*N',V,X,M*S*N',V,dt);
        # S = obj.S_step(solver,X,M*S*N',V,dt);

        # Truncation step
        X,S,V = truncate(solver,X,S,V,ϑ,κ)
        r,_ = size(S);

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function augBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = augBUG(obj);
    return X*S*V',rVec;
end

function augBUG_gpu(obj::DLRAIntegratorSetup)
    dt = Float32(obj.solver.dt)
    Nt = Int(round(obj.solver.tEnd / dt))

    r = obj.r
    ϑ = Float32(obj.ϑ)

    t = 0f0
    k = 0

    psi, phi = SetupIC(obj.solver)

    # SVD in Float32 to initialise GPU factors
    X, s, V = svd(Float32.(psi))
    X = CuMatrix{Float32}(X[:, 1:r])
    V = CuMatrix{Float32}(V[:, 1:r])
    S = CuMatrix{Float32}(Diagonal(s[1:r]))

    m, n = size(X, 1), size(V, 1)

    if should_apply_bc(obj.solver.model)
        # compute_boundary_basis only uses obj.solver internals; K/V args are unused
        _, Wb, r_boundary = compute_boundary_basis(obj.solver, nothing, nothing)
        Wb = CuMatrix{Float32}(Float32.(Wb))
    else
        Wb = CUDA.zeros(Float32, n, 0)
        r_boundary = 0
    end
    κ = obj.κ + r_boundary

    psi = nothing  # free host memory
    phi = nothing

    # Pre-build GPU operator bundle and QoI helper (dispatches on model type)
    gpu_ops  = make_gpu_ops(obj.solver)
    qoi_vecs = make_gpu_qoi(obj.solver)
    rMaxTotal = min(m, n)

    rVec = Dict()

    QoI = Dict()
    while t < Float32(obj.solver.tEnd)
        k += 1
        dt = min(dt, Float32(obj.solver.tEnd) - t)
        QoI["$k"] = Dict()

        aug_r = 2*r + r_boundary

        # K-step
        K = X * S
        K = K_step(gpu_ops, K, V, dt)
        impose_BC!(gpu_ops, K, V)

        Xaug = hcat(CUDA.randn(Float32, m, r_boundary), K, X)
        Xhat = qr(Xaug).Q * CuArray(Matrix{Float32}(I, m, aug_r))
        M    = Xhat' * X

        # L-step
        Lt = S * V'
        Lt = L_step(gpu_ops, X, Lt, dt)

        Vaug = hcat(Wb, Lt', V)
        Vhat = qr(Vaug).Q * CuArray(Matrix{Float32}(I, n, aug_r))
        N    = Vhat' * V

        X, V = Xhat, Vhat

        # S-step
        S = M * S * N'
        S = S_step(gpu_ops, X, S, V, X, S, V, dt)

        # Truncation: stays on GPU — only O(r) singular values cross PCIe
        X, S, V = truncate_gpu(X, S, V, ϑ, rMaxTotal)
        r, _ = size(S)

        t += dt
        if obj.computeQoI
            compute_QoI_gpu(qoi_vecs, obj.solver, obj.solver.model, QoI, X, S, V, k, k == Nt)
        end

        rVec["1", "$k"] = t
        rVec["2", "$k"] = r
        rVec["3", "$k"] = sizeof(X) + sizeof(S) + sizeof(V)
        
    end
    return X, S, V, QoI, rVec;
end

"""
    ThermalSolution

Wraps the output of `augBUG_gpu` so that `Sol_to_FoI` can extract either the
full scalar-flux field *or* the in-situ scalar QoI that were accumulated during
the time integration.

Supported FoI strings (see `Sol_to_FoI` dispatches in uq.jl):
  "ScalarFlux"              – full spatial scalar-flux vector  φ(x)
  Hohlraum-specific:
    "Absorption_GB"         – total absorption in Green/Blue regions
    "Absorption_R"          – total absorption in Red regions
    "Absorption_K"          – total absorption in Black regions
    "MeanBlockAbsorption"   – mean block-wise absorption (44 green blocks)
    "MeanLineAbsorption"    – mean line-wise absorption (44 green lines)
    "TotalMass"             – total scalar mass at final time
  Lattice-specific:
    "Flux_1.5"              – time-integrated flux through boundary at ℓ=1.5
    "Flux_2.5"              – time-integrated flux through boundary at ℓ=2.5
    "Absorption_Blue"       – time-integrated absorption in blue blocks
    "TotalMass"             – total scalar mass at final time
"""
struct ThermalSolution
    phi  ::Vector{Float64}   # scalar flux field (length N)
    QoI  ::Dict              # per-timestep QoI dict from compute_QoI
    Nt   ::Int               # number of timesteps
end

function augBUG_gpu(obj::DLRAIntegratorSetup, sample::Vector{Float64})
    apply_sample!(obj.solver, sample)
    X, S, V, QoI, rVec_dict = augBUG_gpu(obj)
    # Compute scalar flux on CPU (interior cells only): phi = (X*(S*(V'*w)))[interior]
    w        = Float64.(obj.solver.sn.w)
    phi_full = Vector{Float64}(Array(X) * (Matrix(Array(S)) * (Array(V)' * w)))
    phi      = phi_full[obj.solver.model.interior]
    # Convert Dict-style rVec to the 3×Nt matrix that ComputeSample_DLR expects
    Nt     = length(keys(rVec_dict)) ÷ 3
    rVec   = zeros(Float64, 3, Nt)
    for k in 1:Nt
        rVec[1, k] = rVec_dict["1", "$k"]
        rVec[2, k] = rVec_dict["2", "$k"]
        rVec[3, k] = rVec_dict["3", "$k"]
    end
    return ThermalSolution(phi, QoI, Nt), rVec
end

function parBUG(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    g = 0; # Free up memory

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks

    for k = 1:Nt
        # K-step 
        K = X*S;
        K .= obj.K_step(solver,K,V,dt);
        Xtmp,_ = qr([X K]);
        Xtmp = Matrix(Xtmp);
        Xtilde = Xtmp[:,r+1:end];
        Xhat = [X Xtilde];

        # L-step
        Lt = S*V';
        Lt .= obj.L_step(solver,X,Lt,dt);
        Vtmp,_ = qr([V Lt']);
        Vtmp = Matrix(Vtmp);
        Vtilde = Vtmp[:,r+1:end];
        Vhat = [V Vtilde];

        # S-step
        S = obj.S_step(solver,X,S,V,X,S,V,dt);

        Shat = zeros(2*r,2*r);
        Shat[1:r,1:r] .= S;
        Shat[1:r,r+1:end] .= Lt*Vtilde;
        Shat[r+1:end,1:r] .= Xtilde'*K;


        # Truncation step
        X,S,V = truncate(solver,Xhat,Shat,Vhat,ϑ,κ)
        r,_ = size(S);

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function parBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = parBUG(obj);
    return X*S*V',rVec;
end

function parBUG_rejection(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);
    rmax = min(size(g)[1],size(g)[2]);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);
    g = 0;

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt+1); # Array to store the time and ranks
    k = 0;
    t = 0.0;
    prog = Progress(Nt,1)
    while t < Nt*dt
        k += 1;
        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
        # K-step 
        K = X*S;
        K .= obj.K_step(solver,K,V,dt);
        Xtmp,_ = qr([X K]);
        Xtmp = Matrix(Xtmp);
        Xtilde = Xtmp[:,(r+1):end];
        Xhat = [X Xtilde];

        # L-step
        Lt = S*V';
        Lt .= obj.L_step(solver,X,Lt,dt);
        Vtmp,_ = qr([V Lt']);
        Vtmp = Matrix(Vtmp);
        Vtilde = Vtmp[:,(r+1):end];
        Vhat = [V Vtilde];

        # S-step
        S = obj.S_step(solver,X,S,V,X,S,V,dt);

        Shat = zeros(2*r,2*r);
        Shat[1:r,1:r] .= S;
        Shat[1:r,r+1:end] .= Lt*Vtilde;
        Shat[r+1:end,1:r] .= Xtilde'*K;


        # Truncation step
        XUP,SUP,VUP = truncate(solver,Xhat,Shat,Vhat,ϑ,κ)

        # rejection step
        if size(SUP,1) == 2*r && 2*r < rmax
            S = (Xhat'*X)*S*(V'*Vhat)
            X = Xhat;
            V = Vhat;
            r = 2*r;
            k = k-1;
            continue;
        else
            Quasi_S = obj.S_step(solver,X,zeros(size(Xtilde,2),size(Xtilde,2)),V,Xtilde,S,Vtilde,-1.0);

            eta = norm(Quasi_S)

            bound = obj.cη * obj.ϑ * max(1e-11,norm(Shat)) / dt;

            if eta > bound && 2*r < rmax
                println(eta," > ",obj.cη * obj.ϑ * max(1e-7,norm(Shat)) / obj.solver.settings.dt)
                S = (Xhat'*X)*S*(V'*Vhat)
                X = Xhat;
                V = Vhat;
                r = 2*r;
                k = k-1;
                continue;
            end
        end

        X,S,V = XUP,SUP,VUP;
        r,_ = size(S);

        t += dt;
        next!(prog)
    end
    return X,S,V,rVec;
end

function parBUG_rejection(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = parBUG_rejection(obj);
    return X*S*V',rVec;
end

function augBUG_step(obj::DLRAIntegratorSetup,X,S,V,h)
    K = X*S;
    K = obj.K_step(solver,K,V,h);
    Xhat,_ = qr([K X]);
    Xhat = Matrix(Xhat);
    M = Xhat'*X;

    # L-step
    Lt = S*V';
    Lt = obj.L_step(solver,X,Lt,h);
    Vhat,_ = qr([Lt' V]);
    Vhat = Matrix(Vhat);
    N = Vhat'*V;

    X,V = Xhat, Vhat;

    # S-step
    S = obj.S_step(solver,X,M*S*N',V,X,S,V,h);

    # Truncation step
    X,S,V = truncate(solver,X,S,V,obj.ϑ)
    return X,S,V;
end

function fixedrankBUG_step(obj::DLRAIntegratorSetup,X,S,V,h)
    K = X*S;
    K = obj.K_step(solver,K,V,h);
    Xhat,_ = qr(K);
    Xhat = Matrix(Xhat);
    M = Xhat'*X;

    # L-step
    Lt = S*V';
    Lt = obj.L_step(solver,X,Lt,h);
    Vhat,_ = qr(Lt');
    Vhat = Matrix(Vhat);
    N = Vhat'*V;

    X,V = Xhat, Vhat;

    # S-step
    S = obj.S_step(solver,X,M*S*N',V,X,S,V,h);
    return X,S,V;
end

function SOaugBUG(obj::DLRAIntegratorSetup)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks
    for k = 1:Nt
        X0,S0,V0 = X,S,V;
        X,S,V = augBUG_step(obj,X0,S0,V0,dt/2);

        K = obj.K_step(solver,X*S,V,dt);
        K .-= X*S;

        Lt = obj.L_step(solver,X,S*V',dt);
        Lt .-= S*V';

        Xbar,_ = qr([X K])
        Xbar = Matrix(Xbar);

        Vbar,_ = qr([V Lt'])
        Vbar = Matrix(Vbar);

        M = Xbar'*X0;
        N = Vbar'*V0;

        Sbar = obj.S_step(solver,Xbar,M*S0*N',Vbar,Xbar,M*S0*N',Vbar,dt)

        X,S,V = truncate(solver,Xbar,Sbar,Vbar,ϑ,κ)

        r,_ = size(S);

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function SOaugBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = SOaugBUG(obj);
    return X*S*V',rVec;
end

function SOparBUG(obj::DLRAIntegratorSetup,ver::Int=1)
    solver = obj.solver;
    dt = solver.settings.dt;
    Tend = solver.settings.Tend;
    r = obj.r;
    ϑ = obj.ϑ;
    κ = obj.κ;
    t = 0.0;

    g = obj.setupIC(solver);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    rVec = zeros(3,Nt); # Array to store the time and ranks
    for k = 1:Nt
        K = obj.K_step(solver,X*S,V,1.0);
        K .-= X*S;

        Lt = obj.L_step(solver,X,S*V',1.0);
        Lt .-= S*V';

        Xbar,_ = qr([X K])
        Xbar = Matrix(Xbar);
        M = Xbar'*X;

        Vbar,_ = qr([V Lt'])
        Vbar = Matrix(Vbar);
        N = Vbar'*V;

        K = X*S*N';
        K .= obj.K_step(solver,K,Vbar,dt);
        Xtmp,_ = qr([Xbar K]);
        Xtmp = Matrix(Xtmp);
        Xtilde = Xtmp[:,2*r+1:end];
        Xhat = [Xbar Xtilde];

        # L-step
        Lt = M*S*V';
        Lt .= obj.L_step(solver,Xbar,Lt,dt);
        Vtmp,_ = qr([Vbar Lt']);
        Vtmp = Matrix(Vtmp);
        Vtilde = Vtmp[:,2*r+1:end];
        Vhat = [Vbar Vtilde];

        # S-step
        Sbar = obj.S_step(solver,Xbar,M*S*N',Vbar,Xbar,M*S*N',Vbar,dt);

        Shat = zeros(4*r,4*r);
        Shat[1:2*r,1:2*r] = Sbar;
        Shat[1:2*r,2*r+1:end] = Lt*Vtilde;
        Shat[2*r+1:end,1:2*r] = Xtilde'*K;


        # Truncation step
        X,S,V = truncate(solver,Xhat,Shat,Vhat,ϑ,κ)
        r,_ = size(S);

        t = t+dt;

        rVec[1,k] = t;
        rVec[2,k] = r;
        rVec[3,k] = sizeof(X) + sizeof(S) + sizeof(V);
    end
    return X,S,V,rVec;
end

function SOparBUG(obj::DLRAIntegratorSetup,sample::Array{Float64,1})
    obj.solver.sample = sample; 
    X,S,V,rVec = SOparBUG(obj);
    return X*S*V',rVec;
end


function truncate(obj,X::Array{T,2},S::Array{T,2},V::Array{T,2},ϑ::AbstractFloat,κ::Int,fixed::Bool=false) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    m,r = size(X);
    n,_ = size(V);
    rMaxTotal = min(m, n);
    rMinTotal = 2;
    if fixed
        rmax = Int(r/2);
        return  X*U[:, 1:rmax], diagm(D[1:rmax]), V*W[:, 1:rmax];
    else
        if κ == 0
            U,D,W = svd(Matrix(S));
    
            rmax = -1;
            
            # adaptIndex = 1;

            tmp = 0.0;
            tol = ϑ*norm(D);
            rmax = Int(floor(size(D,1)/2));

            for j=1:2*rmax
                tmp = sqrt(sum(D[j:2*rmax]).^2);
                if tmp < tol
                    rmax = j;
                    break;
                end
            end

            # if 2*r was actually not enough move to highest possible rank
            if rmax == -1
                rmax = rMaxTotal;
            end

            rmax = min(rmax,rMaxTotal);
            rmax = max(rmax,rMinTotal);
        
            return  X*U[:, 1:rmax], diagm(D[1:rmax]), V*W[:, 1:rmax];
        else
            

            # Conservative truncation
            Khat = X * S;
            Khat_ap, Khat_rem = Khat[:,1:κ], Khat[:,κ+1:end]; # Splitting Khat into basis required for Ap and remaining vectors
            Vap, Vrem = V[:,1:κ], V[:,κ+1:end]; # Splitting Khat into basis required for Ap and remaining vectors

            Xhrem, Shrem = qr(Khat_rem);
            Xhrem = Matrix(Xhrem);

            U,D,W = svd(Matrix(Shrem));
            U = Matrix(U);
            W = Matrix(W);

            rmax = -1
            tmp = 0.0;

            tol = ϑ * norm(D);

            # Truncating the rank
            for i = 1:r-κ
                tmp = sqrt(sum(sigma[i:end].^2));
                if tmp < tol
                    rmax = i
                    break
                end
            end

            rmax = min(rmax,rMaxTotal);
            rmax = max(rmax,rMinTotal);

            if rmax == -1
                rmax = rmaxTotal;
            end

            Uhat = U[:,1:rmax];
            What = W[:,1:rmax];
            sigma_hat = diagm(sigma[1:rmax]);

            W1 = Vrem * What;
            Xrem = Xhrem * Uhat;

            V = [Vap W1];
            Xap, Sap = qr(Khat_ap);
            Xap = Matrix(Xap);
            X, R2 = qr([Xap Xrem]);
            X = Matrix(X);

            S = zeros(rmax+m,rmax+m);
            S[1:m,1:m] = Sap;
            S[m+1:end,m+1:end] = sigma_hat;
            S .= R2 * S;
            return  X, S, V;
        end
    end
end