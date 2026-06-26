__precompile__

using ProgressMeter
using ProgressBars
using LinearAlgebra
using FastGaussQuadrature, LegendrePolynomials
using PyCall
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using SpecialFunctions

mutable struct solver
    # Spatial grid of cell vertices
    x::Array{Float64,1};
    xMid::Array{Float64,1};

    gridSize::Array{Float64,1}; # Required for setting QoI in MLMC
    gridWidth::Array{Float64,1}; # Required for computing norms in MLMC

    # Solver settings
    settings::settings;

    ## Angular discretisation

    # Pn discretisation 
    A::SparseMatrixCSC{Float32,Int64};
    absA::SparseMatrixCSC{Float32,Int64};
    G::SparseMatrixCSC{Float32,Int64};

    # Stencil matrices for spatial discretisation
    Dx::SparseMatrixCSC{Float32,Int64};
    Dxx::SparseMatrixCSC{Float32,Int64};

    # Physical parameters
    sigmaA::Float64;
    sigmaS::Float64;

    # Uncertainty for initial condition
    sample::Array{Float64,1};
    uncertParam::String;
    # Constructor
    function solver(settings)
        x = settings.x;
        xMid = settings.xMid;

        nx = settings.Nx;
        nxC = settings.NxC;

        gridSize = [nx];

        # Setting up the matrices for the Pn solver
        nPN = settings.nPN; # total number of Legendre polynomials used
        gamma = zeros(nPN+1); # vector with norms of the Legendre polynomials

        for i = 1:nPN+1
            gamma[i] = 2/(2*(i-1) + 1);
        end

        A = zeros(Float64,nPN+1,nPN+1); # reduced Flux matrix for the micro equation
        a_norm = zeros(Float64,nPN+1);

        for i = 1:nPN+1
            a_norm[i] = i/(sqrt((2i-1)*(2i+1)));
        end

        A = Tridiagonal(a_norm[1:end-1],zeros(nPN+1),a_norm[1:end-1]); # Full flux matrix
        # Ix = []; Jx = []; Vx = [];
        # for i = 1:nPN+1
        #     if i+1 <= nPN+1
        #         push!(Ix, i);
        #         push!(Jx, i+1);
        #         push!(Vx, a_norm[i]);
        #     end
        #     if i-1 >= 1
        #         push!(Ix, i);
        #         push!(Jx, i-1);
        #         push!(Vx, a_norm[i]);
        #     end
        # end
        # A = sparse(Ix,Jx,Float32.(Vx),nPN+1,nPN+1);
        # println(A[1,2], Float32.(Af[1,2]));
        S = eigvals(Matrix(A));
        V = eigvecs(Matrix(A));
        absA = V*abs.(diagm(S))*inv(V);

        # idx = findall(abs.(absA) .> 1e-10);
        # Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = absA[idx];
        # absA = sparse(Ix,Jx,vals,nPN+1,nPN+1);

        Ix = []; Jx = []; Vx = [];
        for i = 1:nPN+1
            if i == 1#i != 1
                push!(Ix, i);
                push!(Jx, i);
                push!(Vx, -1.0); # push!(Vx, 1.0/sqrt(2));
            end 
        end
        G = sparse(Ix,Jx,Vx,nPN+1,nPN+1); # Reduced scattering matrix
        # G = zeros(nPN+1,nPN+1);
        # G[1,1] = -1.0;

        dx = settings.dx;
        gridWidth = [dx];

        # Stencil matrices for the Pn solver
        Dx = spzeros(Float64,nx,nx);
        Dxx = spzeros(Float64,nx,nx);
        Ix = []; Jx = []; Vx = []; 
        for i = 1:nx
            if i+1 <= nx
                push!(Ix, i);
                push!(Jx, i+1);
                push!(Vx, 1.0/(2.0*dx));
            end
            if i-1 >= 1
                push!(Ix, i);
                push!(Jx, i-1);
                push!(Vx, -1.0/(2.0*dx));
            end
        end
        Dx = sparse(Ix,Jx,Vx,nx,nx); 
        Ix = []; Jx = []; Vx = []; 
        for i = 1:nx
            push!(Ix, i);
            push!(Jx, i);
            push!(Vx, -1.0/(dx));
            if i+1 <= nx
                push!(Ix, i);
                push!(Jx, i+1);
                push!(Vx, 1.0/(2.0*dx));
            end
            if i-1 >= 1
                push!(Ix, i);
                push!(Jx, i-1);
                push!(Vx, 1.0/(2.0*dx));
            end
        end

        Dxx = sparse(Ix,Jx,Vx,nx,nx); 
        # Dx = Tridiagonal(-ones(nx-1)./2.0/dx, zeros(nx), ones(nx-1)./2.0/dx);
        # Dxx = Tridiagonal(ones(nx-1)./2.0/dx, -ones(nx)./dx, ones(nx-1)./2.0/dx);

        sample = [1.0,0.0]; # Placeholder for the sample parameter

        uncertParam = "1";

        new(x,xMid,gridSize,gridWidth,settings,A,absA,G,Dx,Dxx,settings.sigmaA,settings.sigmaS,sample,uncertParam);
    end
 end


 function setupIC(obj::solver)
    g = zeros(obj.settings.Nx,obj.settings.nPN+1);
    g[:,1] = IC(obj.settings,obj.sample,obj.uncertParam);
    if obj.settings.problem == "GaussianPulse"
        obj.settings.sigmaS = (1.0+obj.sample[1]/10);
        # obj.settings.sigmaA = 1-obj.settings.sigmaS;
    end
    return g;
 end
 
function K_step(obj,K,V,dt)
    VGV = V'*obj.G*V;
    VAV = V'*obj.A'*V;
    VabsAV = V'*obj.absA*V;
    K .= K .+ dt*(-obj.Dx*K*VAV + obj.Dxx*K*VabsAV - obj.settings.sigmaS*K*VGV - obj.settings.sigmaA*K);
    return K;
end

function L_step(obj,X,Lt,dt)
    XDxX = X'*obj.Dx*X;
    XDxxX = X'*obj.Dxx*X;
    Lt .= Lt .+ dt*(-XDxX*Lt*obj.A' + XDxxX*Lt*obj.absA' - obj.settings.sigmaS*Lt*obj.G - obj.settings.sigmaA*Lt);
    return Lt;
end

function S_step(obj,X,S,V,Xh,Sh,Vh,dt)
    XDxX = Xh'*obj.Dx*X;
    XDxxX = Xh'*obj.Dxx*X;
    VGV = V'*obj.G*Vh;
    VAV = V'*obj.A'*Vh;
    VabsAV = V'*obj.absA*Vh;
    S .= S .+ dt*(-XDxX*Sh*VAV + XDxxX*Sh*VabsAV - obj.settings.sigmaS*Sh*VGV - obj.settings.sigmaA*Sh);
    return S;
end

function pre_step()
    return nothing
end

function post_step()
    return nothing
end


function solveFullProblem(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    A = obj.A;
    absA = obj.absA;
    G = obj.G;
    Dx = obj.Dx;
    Dxx = obj.Dxx;

    g = setupIC(obj);
    g0 = zeros(size(g));
    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 
    for k = 1:Nt
        g0 .= g
        g .= g .- dt.*(Dx*g0*Transpose(A));
        g .= g .+ dt.*(Dxx*g0*Transpose(absA));
        g .= g .- dt.*(obj.settings.sigmaS*g0*G);
        g .= g .- dt.*(obj.settings.sigmaA*g0);

        # g .= g .+ dt.*(-Dx*g*A' + Dxx*g*absA' - Float32(obj.settings.sigmaS).*g*G - Float32(obj.settings.sigmaA).*g);
    end

    return g;

    end

function solveFullProblem(sample::Float64,Nx::Int64,N::Int)
    s = settings1D(Nx,N);
    solver = solver(s);

    solver.sample = sample;

    g = solveFullProblem(solver);

    return g
    end

function FullProblem(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g = solveFullProblem(obj);
    return g,zeros(2,3);
    end

    function CUsolveFullProblem(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    A = CuSparseMatrixCSC(obj.A);
    absA = CuSparseMatrixCSC(obj.absA);
    G = CuSparseMatrixCSC(obj.G);
    Dx = CuSparseMatrixCSC(obj.Dx);
    Dxx = CuSparseMatrixCSC(obj.Dxx);

    g = cu(setupIC(obj));
    g0 = zeros(size(g));

    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 
    dt = Float32(dt); # Ensure dt is Float32 for CUDA compatibility
    for k = 1:Nt
        g = g + dt*(-Dx*g*A' + Dxx*g*absA' - Float32(obj.settings.sigmaS)*g*G - Float32(obj.settings.sigmaA)*g);
    end

    return g;

end

function CUsolveFullProblem(sample::Float64,Nx::Int64,N::Int)
    s = settings1D(Nx,N);
    solver = solver(s);

    solver.sample = sample;

    g = CUsolveFullProblem(solver);

    return g
end

function CUsolveFullProblem(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g = CUsolveFullProblem(obj);
    return g
end


function solvefixedrankBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;

    # A = obj.A;
    # absA = obj.absA;
    # G = obj.G;

    # Dx = obj.Dx;
    # Dxx = obj.Dxx;

    g = setupIC(obj);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    VGV = zeros(Float64,r,r);
    VAV = zeros(Float64,r,r);
    VabsAV = zeros(Float64,r,r);
    
    XDxX = zeros(Float64,r,r);
    XDxxX = zeros(Float64,r,r);

    K = zeros(size(X));
    K0 = zeros(size(X));
    Lt = zeros(size(transpose(V)));
    Lt0 = zeros(size(transpose(V)));
    S0 = zeros(size(S));

    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt

        ## K-step
        K .= X*S;
        K0 .= K;
        VGV .= transpose(V)*obj.G*V;
        VAV .= transpose(V)*transpose(obj.A)*V;
        VabsAV .= transpose(V)*transpose(obj.absA)*V;

        # K .= K .+ dt*(-Dx*K*VAV + Dxx*K*VabsAV - obj.settings.sigmaS*K*VGV - obj.settings.sigmaA*K);
        K .= K .- dt.*obj.Dx*K0*VAV;
        K .= K .+ dt.*obj.Dxx*K0*VabsAV;
        K .= K .- dt.*obj.settings.sigmaS*K0*VGV;
        K .= K .- dt.*obj.settings.sigmaA*K0;

        X1,_ = qr(K);
        X1 = Matrix(X1);
        M = transpose(X1)*X;
        
        ## L-step
        Lt .= S*transpose(V);
        Lt0 .= Lt;
        XDxX .= transpose(X)*obj.Dx*X;
        XDxxX .= transpose(X)*obj.Dxx*X;
        
        # Lt .= Lt .+ dt*(-XDxX*Lt*Transpose(A) + XDxxX*Lt*Transpose(absA) - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);
        Lt .= Lt .- dt.*XDxX*Lt0*Transpose(obj.A);
        Lt .= Lt .+ dt.*XDxxX*Lt0*Transpose(obj.absA);
        Lt .= Lt .- dt.*obj.settings.sigmaS*Lt0*obj.G;
        Lt .= Lt .- dt.*obj.settings.sigmaA*Lt0;

        V1,_ = qr(transpose(Lt)); 
        V1 = Matrix(V1)
        N = transpose(V1)*V;

        X,V = X1, V1;

        ## S-step
        S .= M*S*transpose(N);
        S0 .= S;
        XDxX .= transpose(X)*obj.Dx*X;
        XDxxX .= transpose(X)*obj.Dxx*X;
        VGV .= transpose(V)*obj.G*V;
        VAV .= transpose(V)*transpose(obj.A)*V;
        VabsAV .= transpose(V)*transpose(obj.absA)*V;
        
        # S .= S .+ dt*(-XDxX*S*VAV + XDxxX*S*VabsAV - obj.settings.sigmaS*S*VGV - obj.settings.sigmaA*S)

        S .= S .- dt.*XDxX*S0*VAV;
        S .= S .+ dt.*XDxxX*S0*VabsAV;
        S .= S .- dt.*obj.settings.sigmaS*S0*VGV;
        S .= S .- dt.*obj.settings.sigmaA*S0;

        t = t+dt;
    end
    return X*S*transpose(V);

end

function solvefixedrankBUG(sample::Float64,Nx::Int64,N::Int,r::Int)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.sample = sample;

    g = solvefixedrankBUG(solver);

    return g
end

function solvefixedrankBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample[1]; # Need to change this to support multiple samples
    g = solvefixedrankBUG(obj);
    return g
end

function CUsolvefixedrankBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;
    obj.settings.sigmaS = Float32(obj.settings.sigmaS);
    obj.settings.sigmaA = Float32(obj.settings.sigmaA);

    A = CuSparseMatrixCSC(obj.A);
    AT = transpose(A);
    absA = CuSparseMatrixCSC(obj.absA);
    absAT = transpose(absA);
    G = CuSparseMatrixCSC(obj.G);
    Dx = CuSparseMatrixCSC(obj.Dx);
    Dxx = CuSparseMatrixCSC(obj.Dxx);

    g = Float32.(setupIC(obj));

    X,s,V = svd(g);
    X = cu(X[:,1:r]);
    V = cu(V[:,1:r]);
    S = cu(diagm(s[1:r]));

    X1 = CUDA.zeros(Float32,size(X));
    V1 = CUDA.zeros(Float32,size(V));

    VGV = CUDA.zeros(Float32,r,r);
    VAV = CUDA.zeros(Float32,r,r);
    VabsAV = CUDA.zeros(Float32,r,r);
    
    XDxX = CUDA.zeros(Float32,r,r);
    XDxxX = CUDA.zeros(Float32,r,r);

    M = CUDA.zeros(Float32,r,r);
    N = CUDA.zeros(Float32,r,r);

    K = CUDA.zeros(Float32,size(X));
    Lt = CUDA.zeros(Float32,size(transpose(V)));

    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Float32(Tend/Nt); # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt

        ## K-step
        K .= X*S;
        VGV .= transpose(V)*G*V;
        VAV .= transpose(V)*AT*V;
        VabsAV .= transpose(V)*absAT*V;

        K .= K .+ dt .*(-Dx*K*VAV + Dxx*K*VabsAV - obj.settings.sigmaS*K*VGV - obj.settings.sigmaA*K);


        X1,_ = CUDA.qr(K);
        X1 = cu(X1);
        M .= transpose(X1)*X;
        
        ## L-step
        Lt .= S*transpose(V);
        XDxX .= transpose(X)*Dx*X;
        XDxxX .= transpose(X)*Dxx*X;
        
        Lt .= Lt .+ dt .*(-XDxX*Lt*AT + XDxxX*Lt*absAT - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);

        V1,_ = CUDA.qr(Lt'); 
        V1 = cu(V1);
        N .= transpose(V1)*V;

        X,V = X1, V1;

        ## S-step
        S .= M*S*transpose(N);
        XDxX .= transpose(X)*Dx*X;
        XDxxX .= transpose(X)*Dxx*X;
        VGV .= transpose(V)*G*V;
        VAV .= transpose(V)*AT*V;
        VabsAV .= transpose(V)*absAT*V;
        
        S .= S .+ dt .*(-XDxX*S*VAV + XDxxX*S*VabsAV - obj.settings.sigmaS*S*VGV - obj.settings.sigmaA*S)

        t = t+dt;
    end
    return X*S*transpose(V);

end

function CUsolvefixedrankBUG(sample::Float64,Nx::Int64,N::Int,r::Int)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.sample = sample;

    g = CUsolvefixedrankBUG(solver);

    return g
end

function CUsolvefixedrankBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g = CUsolvefixedrankBUG(obj);
    return g
end

function solvefixedAugBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;

    # A = obj.A;
    # absA = obj.absA;
    # G = obj.G;

    # Dx = obj.Dx;
    # Dxx = obj.Dxx;

    g = setupIC(obj);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    VGVr = zeros(Float64,r,r);
    VAVr = zeros(Float64,r,r);
    VabsAVr = zeros(Float64,r,r);
    
    XDxXr = zeros(Float64,r,r);
    XDxxXr = zeros(Float64,r,r);

    VGV2r = zeros(Float64,2*r,2*r);
    VAV2r = zeros(Float64,2*r,2*r);
    VabsAV2r = zeros(Float64,2*r,2*r);
    
    XDxX2r = zeros(Float64,2*r,2*r);
    XDxxX2r = zeros(Float64,2*r,2*r);

    K = zeros(size(X));
    K0 = zeros(size(X));
    Lt = zeros(size(transpose(V)));
    Lt0 = zeros(size(transpose(V)));
    S0 = zeros(2*r,2*r);
    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt

        ## K-step

        K .= X*S;
        K0 .= K;
        VGVr .= transpose(V)*obj.G*V;
        VAVr .= transpose(V)*transpose(obj.A)*V;
        VabsAVr .= transpose(V)*transpose(obj.absA)*V;

        # K .= K .+ dt*(-obj.Dx*K*VAVr + obj.Dxx*K*VabsAVr - obj.settings.sigmaS*K*VGVr - obj.settings.sigmaA*K);
        K .= K .- dt.*obj.Dx*K0*VAVr;
        K .= K .+ dt.*obj.Dxx*K0*VabsAVr;
        K .= K .- dt.*obj.settings.sigmaS*K0*VGVr;
        K .= K .- dt.*obj.settings.sigmaA*K0;

        Xhat,_ = qr([X K]); 
        Xhat = Matrix(Xhat);
        M = transpose(Xhat)*X;
        
        
        ## L-step
        Lt .= S*transpose(V);
        Lt0 .= Lt;
        XDxXr .= transpose(X)*obj.Dx*X;
        XDxxXr .= transpose(X)*obj.Dxx*X;
        
        # Lt .= Lt .+ dt*(-XDxXr*Lt*Transpose(obj.A) + XDxxXr*Lt*Transpose(obj.absA) - obj.settings.sigmaS*Lt*obj.G - obj.settings.sigmaA*Lt);
        Lt .= Lt .- dt.*XDxXr*Lt0*Transpose(obj.A);
        Lt .= Lt .+ dt.*XDxxXr*Lt0*Transpose(obj.absA);
        Lt .= Lt .- dt.*obj.settings.sigmaS*Lt0*obj.G;
        Lt .= Lt .- dt.*obj.settings.sigmaA*Lt0;

        Vhat,_ = qr([transpose(Lt) V]);
        Vhat = Matrix(Vhat);
        N = transpose(Vhat)*V;

        # X,V = Xhat, Vhat;

        ## S-step
        S = M*S*transpose(N);
        S0 = S;
        XDxX2r .= transpose(Xhat)*obj.Dx*Xhat;
        XDxxX2r .= transpose(Xhat)*obj.Dxx*Xhat;
        VGV2r .= transpose(Vhat)*obj.G*Vhat;
        VAV2r .= transpose(Vhat)*transpose(obj.A)*Vhat;
        VabsAV2r .= transpose(Vhat)*transpose(obj.absA)*Vhat;

        # S .= S .+ dt.*(-XDxX2r*S*VAV2r .+ XDxxX2r*S*VabsAV2r .- obj.settings.sigmaS*S*VGV2r .- obj.settings.sigmaA*S)
        
        S .= S .- dt.*XDxX2r*S0*VAV2r;
        S .= S .+ dt.*XDxxX2r*S0*VabsAV2r;
        S .= S .- dt.*obj.settings.sigmaS*S0*VGV2r;
        S .= S .- dt.*obj.settings.sigmaA*S0;

        P,sig,Q = svd(S);

        X .= Xhat*P[:,1:r];
        V .= Vhat*Q[:,1:r];
        S = diagm(sig[1:r]);

        t = t+dt;
    end
    return X*S*transpose(V);

end

function solvefixedAugBUG(sample::Float64,Nx::Int64,N::Int,r::Int)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.sample = sample;

    g = solvefixedAugBUG(solver);

    return g
end

function solvefixedAugBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g = solvefixedAugBUG(obj);
    return g
end

function CUsolvefixedAugBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;
    obj.settings.sigmaS = Float32(obj.settings.sigmaS);
    obj.settings.sigmaA = Float32(obj.settings.sigmaA);

    A = CuSparseMatrixCSC(obj.A);
    AT = transpose(A);
    absA = CuSparseMatrixCSC(obj.absA);
    absAT = transpose(absA);
    G = CuSparseMatrixCSC(obj.G);
    Dx = CuSparseMatrixCSC(obj.Dx);
    Dxx = CuSparseMatrixCSC(obj.Dxx);

    g = Float32.(setupIC(obj));

    X,s,V = svd(g);
    X = cu(X[:,1:r]);
    V = cu(V[:,1:r]);
    S = cu(diagm(s[1:r]));

    Xhat = CUDA.zeros(Float32,size(X));
    Vhat = CUDA.zeros(Float32,size(V));

    VGVr = CUDA.zeros(Float32,r,r);
    VAVr = CUDA.zeros(Float32,r,r);
    VabsAVr = CUDA.zeros(Float32,r,r);
    
    XDxXr = CUDA.zeros(Float32,r,r);
    XDxxXr = CUDA.zeros(Float32,r,r);

    VGV2r = CUDA.zeros(Float32,2*r,2*r);
    VAV2r = CUDA.zeros(Float32,2*r,2*r);
    VabsAV2r = CUDA.zeros(Float32,2*r,2*r);
    
    XDxX2r = CUDA.zeros(Float32,2*r,2*r);
    XDxxX2r = CUDA.zeros(Float32,2*r,2*r);

    M = CUDA.zeros(Float32,2*r,r);
    N = CUDA.zeros(Float32,2*r,r);

    K = CUDA.zeros(Float32,size(X));
    Lt = CUDA.zeros(Float32,size(transpose(V)));

    
    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Float32(Tend/Nt); # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt
        ## K-step

        K .= X*S;
        VGVr .= transpose(V)*G*V;
        VAVr .= transpose(V)*AT*V;
        VabsAVr .= transpose(V)*absAT*V;

        K .= K .+ dt .*(-Dx*K*VAVr + Dxx*K*VabsAVr - obj.settings.sigmaS*K*VGVr - obj.settings.sigmaA*K);

        Xhat,_ = CUDA.qr([X K]); 
        Xhat = cu(Xhat);
        M .= transpose(Xhat)*X;
        
        
        ## L-step
        Lt .= S*transpose(V);
        XDxXr .= transpose(X)*Dx*X;
        XDxxXr .= transpose(X)*Dxx*X;
        
        Lt .= Lt .+ dt .*(-XDxXr*Lt*AT + XDxxXr*Lt*absAT - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);

        Vhat,_ = CUDA.qr([transpose(Lt) V]);
        Vhat = cu(Vhat);
        N .= transpose(Vhat)*V;

        X,V = Xhat, Vhat;

        ## S-step
        S = M*S*transpose(N);
        XDxX2r .= transpose(Xhat)*Dx*Xhat;
        XDxxX2r .= transpose(Xhat)*Dxx*Xhat;
        VGV2r .= transpose(Vhat)*G*Vhat;
        VAV2r .= transpose(Vhat)*AT*Vhat;
        VabsAV2r .= transpose(Vhat)*absAT*Vhat;

        S .= S .+ dt .*(-XDxX2r*S*VAV2r .+ XDxxX2r*S*VabsAV2r .- obj.settings.sigmaS*S*VGV2r .- obj.settings.sigmaA*S)
        
        P,sig,Q = svd(S);

        X = Xhat*cu(P[:,1:r]);
        V = Vhat*cu(Q[:,1:r]);
        S = cu(diagm(sig[1:r]));

        t = t+dt;
    end
    return X*S*transpose(V);

end

function CUsolvefixedAugBUG(sample::Float64,Nx::Int64,N::Int,r::Int)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.sample = sample;

    g = CUsolvefixedAugBUG(solver);

    return g
end

function CUsolvefixedAugBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g = CUsolvefixedAugBUG(obj);
    return g
end



function solveAugBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;

    A =obj.A;
    At = transpose(A);
    absA = obj.absA;
    absAT = transpose(absA);
    G = obj.G;
    Dx = obj.Dx;
    Dxx = obj.Dxx;


    g = setupIC(obj);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    # S = CUDA.zeros(Float32, r, r);    
    # for i = 1:r
    #     S[i,i] = s[i];
    # end
    S = diagm(s[1:r]);

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    r = obj.settings.r;
    rVec = zeros(2,Nt)
    
    for k = 1:Nt

        ## K-step
        K = X*S;
        VGV = transpose(V)*G*V;
        VAV = transpose(V)*At*V;
        VabsAV = transpose(V)*absAT*V;

        K = K .+ dt*(-Dx*K*VAV + Dxx*K*VabsAV - obj.settings.sigmaS*K*VGV - obj.settings.sigmaA*K);
        Xhat,_,_ = svd!([X K]); 
        M = transpose(Xhat)*X;

        ## L-step
        Lt = S*transpose(V);
        XDxX = transpose(X)*Dx*X;
        XDxxX = transpose(X)*Dxx*X;
        
        Lt = Lt .+ dt*(-XDxX*Lt*At + XDxxX*Lt*absAT - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);
        Vhat,_,_ = svd!([transpose(Lt) V]);
        N = transpose(Vhat)*V;
        X,V = Xhat, Vhat;

        ## S-ste
        S = M*S*transpose(N);
        XDxX = transpose(Xhat)*Dx*Xhat;
        XDxxX = transpose(Xhat)*Dxx*Xhat;
        VGV = transpose(Vhat)*G*Vhat;
        VAV = transpose(Vhat)*At*Vhat;
        VabsAV = transpose(Vhat)*absAT*Vhat;
        
        S = S .+ dt*(-XDxX*S*VAV + XDxxX*S*VabsAV - obj.settings.sigmaS*S*VGV - obj.settings.sigmaA*S)


    
        X,S,V = truncate(obj,X,S,V)
        # time_profile[4,k] = t1;
        r = size(S,1)
        rVec[1,k] = t;
        rVec[2,k] = r;
        t = t+dt;
    end
    return X*S*transpose(V),rVec;

end


function solveAugBUG(sample::Float64,Nx::Int64,N::Int,tol::Float64)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.epsAdapt = tol;

    solver.sample = sample;

    g = solveAugBUG(solver);

    return g
end

function solveAugBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g,rVec = solveAugBUG(obj);
    return g,rVec;
end

function CUsolveAugBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;
    obj.settings.sigmaS = Float32(obj.settings.sigmaS);
    obj.settings.sigmaA = Float32(obj.settings.sigmaA);

    A = CuSparseMatrixCSC(obj.A);
    AT = transpose(A);
    absA = CuSparseMatrixCSC(obj.absA);
    absAT = transpose(absA);
    G = CuSparseMatrixCSC(obj.G);
    Dx = CuSparseMatrixCSC(obj.Dx);
    Dxx = CuSparseMatrixCSC(obj.Dxx);


    g = Float32.(setupIC(obj));

    X,s,V = svd(g);
    X = cu(X[:,1:r]);
    V = cu(V[:,1:r]);
    S = cu(diagm(s[1:r]));

    Nt = Int(round(Tend/dt)); # Computing the number of steps required 
    dt = Float32(Tend/Nt); # Adjusting the step size 

    r = obj.settings.r;
    rVec = zeros(2,Nt)
    for k = 1:Nt
        # Xhat = CUDA.zeros(Float32,size(X));
        # Vhat = CUDA.zeros(Float32,size(V));

        # VGVr = CUDA.zeros(Float32,r,r);
        # VAVr = CUDA.zeros(Float32,r,r);
        # VabsAVr = CUDA.zeros(Float32,r,r);
        
        # XDxXr = CUDA.zeros(Float32,r,r);
        # XDxxXr = CUDA.zeros(Float32,r,r);

        # VGV2r = CUDA.zeros(Float32,2*r,2*r);
        # VAV2r = CUDA.zeros(Float32,2*r,2*r);
        # VabsAV2r = CUDA.zeros(Float32,2*r,2*r);
        
        # XDxX2r = CUDA.zeros(Float32,2*r,2*r);
        # XDxxX2r = CUDA.zeros(Float32,2*r,2*r);

        # M = CUDA.zeros(Float32,2*r,r);
        # N = CUDA.zeros(Float32,2*r,r);

        # K = CUDA.zeros(Float32,size(X));
        # Lt = CUDA.zeros(Float32,size(transpose(V)));


        ## K-step
        K = X * S;
        VGVr = (transpose(V)*G*V);
        VAVr = (transpose(V)*AT*V);
        VabsAVr = (transpose(V)*absAT*V);

        K = K + dt *(-Dx*K*VAVr + Dxx*K*VabsAVr - obj.settings.sigmaS*K*VGVr - obj.settings.sigmaA*K);
        Xhat,_ = CUDA.qr([X K]); 
        Xhat = cu(Xhat);
        M = (transpose(Xhat)*X);

        ## L-step

        Lt = S*transpose(V);
        XDxXr = (transpose(X)*Dx*X);
        XDxxXr = (transpose(X)*Dxx*X);
        
        Lt = Lt + dt *(-XDxXr*Lt*AT + XDxxXr*Lt*absAT - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);
        Vhat,_ = CUDA.qr([transpose(Lt) V]);
        Vhat = cu(Vhat);
        N = (transpose(Vhat)*V);

        ## S-step
        S = M*S*transpose(N);
        XDxX2r = (transpose(Xhat)*Dx*Xhat);
        XDxxX2r = (transpose(Xhat)*Dxx*Xhat);
        VGV2r = (transpose(Vhat)*G*Vhat);
        VAV2r = (transpose(Vhat)*AT*Vhat);
        VabsAV2r = (transpose(Vhat)*absAT*Vhat);
        
        S = S + dt *(-XDxX2r*S*VAV2r + XDxxX2r*S*VabsAV2r - obj.settings.sigmaS*S*VGV2r - obj.settings.sigmaA*S)

        P,sig,Q = CUDA.svd(S);
        sig1 = Array(sig);
        # println("SVD time: ", t)

        rmax = -1;
        rMaxTotal = Int(round(min(obj.settings.Nx, obj.settings.nPN+1)/2));
        rMinTotal = 2;
        adaptIndex = 1;

        tmp = 0.0;
        tol = obj.settings.epsAdapt*norm(sig1);
        rmax = Int(floor(length(sig1)/2));

        for j=1:2*rmax
            tmp = sqrt(sum(sig1[j:2*rmax]).^2);
            if tmp < tol
                rmax = j;
                break;
            end
        end
    
        # println("Adaptation time: ", t)

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,rMinTotal);

        # return rank
        ermax = zeros(Float32, length(sig1),rmax);
        for i = 1:rmax
            ermax[i,i] = 1.0;
        end
        ermax = cu(ermax);
        P1 = P*ermax;
        Q1 = Q*ermax;

        X = Xhat*P1;
        X = cu(X);

        S = cu(diagm(sig1[1:rmax]));

        V = Vhat*Q1;
        V = cu(V);

        # r = size(S,1)
        r = rmax;
        rVec[1,k] = t;
        rVec[2,k] = rmax;
        t = t+dt;
    end
    return X*S*transpose(V),rVec;

end


function CUsolveAugBUG(sample::Float64,Nx::Int64,N::Int,tol::Float64)
    s = settings1D(Nx,N);
    solver = solver(s);
    solver.settings.epsAdapt = tol;

    solver.sample = sample;

    g = CUsolveAugBUG(solver);

    return g
end

function CUsolveAugBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample; # Need to change this to support multiple samples
    g,rVec = CUsolveAugBUG(obj);
    return g,rVec;
end

function truncate(obj::solver,X::Array{T,2},S::Array{T,2},W::Array{T,2}) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(Matrix(S));
    rmax = -1;
    rMaxTotal = min(obj.settings.Nx, obj.settings.nPN+1);
    rMinTotal = 2;
    # adaptIndex = 1;

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D);
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

    # return rank
    return  X*U[:, 1:rmax], diagm(D[1:rmax]), W*V[:, 1:rmax];
end

function CUtruncate(obj::solver,X::CuArray{Float32,2},S::CuArray{Float32,2},V::CuArray{Float32,2}) # where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:

    P,sig,Q = svd(S);
    sig = Array(sig);
    # println("SVD time: ", t)

    rmax = -1;
    rMaxTotal = Int(round(min(obj.settings.Nx, obj.settings.nPN+1)/2));
    rMinTotal = 2;
    adaptIndex = 1;

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(sig);
    rmax = Int(floor(length(sig)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(sig[j:2*rmax]).^2);
        if tmp < tol
            rmax = j;
            break;
        end
    end
  
    # println("Adaptation time: ", t)

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rMinTotal);

    # return rank
    ermax = zeros(Float32, length(sig),rmax);
    for i = 1:rmax
        ermax[i,i] = 1.0;
    end
    ermax = cu(ermax);
    P1 = P*ermax;
    Q1 = Q*ermax;

    X1 = X*P1;
    X1 = cu(X1);

    S = diagm(sig[1:rmax]);
    S = cu(S);

    V1 = V*Q1;
    V1 = cu(V1);

    # println("Truncation time: ", t)
    return X1,S,V1;
end


