__precompile__

using ProgressMeter
using LinearAlgebra
using FastGaussQuadrature
using NPZ
using LegendrePolynomials

mutable struct solver
    # spatial grid of cell interfaces
    x::Array{Float64,1};

    gridSize::Array{Float64,1}; # Required for setting QoI in MLMC
    gridWidth::Array{Float64,1}; # Required for computing norms in MLMC


    # Solver settings
    settings::settings;

    # preallocate memory for performance
    outRhsSW::Array{Float64,2};
    outRhsCorr::Array{Float64,2};
    outRhs::Array{Float64,2};

    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # flux matrix
    A::Array{Float64,2};
    # source matrix 1
    B::Array{Float64,2};
    # source matrix 2
    C::Array{Float64,2};

    r::Int

    # basis functions for velocity profile
    phi0::Array{Float64,1};
    phi0W::Array{Float64,1};
    phi::Array{Float64,2};
    phiW::Array{Float64,2};

    
    # Paramenters related to UQ
    uncertParam::String;
    sample::Array{Float64,1};


    # constructor
    function solver(settings)
        x = settings.x;

        gridSize = [length(settings.xMid)];
        gridWidth = [settings.dx];

        outRhsSW = zeros(settings.NCells,2);
        outRhsCorr = zeros(settings.NCells,settings.N-2);
        outRhs = zeros(settings.NCells,settings.N);

        # setup gamma vector
        gamma = zeros(settings.N);
        for i = 1:settings.N
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end

        # setup flux matrix
        N = settings.N
        A = zeros(N-2,N-2)

        for i=1:(N-3)
            n = i+1
            A[i,i+1] = (n+1)/(2*n+1);
            A[i+1,i]= (n-1)/(2*n-1);
        end

        # update dt with correct maximal speed lmax
        #lmax = maximum(abs.(eigvals(A)));
        #settings.dt = settings.dx*settings.cfl/lmax;

        C = zeros(settings.N-1,settings.N-1)
        nu = settings.nu
        lambda = settings.lambda
        for i = 1:(settings.N-1)
            C[i,:] .= -(2*(i+1)+1)*nu/lambda
        end

        B = zeros(settings.N-1,settings.N-1)

        for i = 1:(settings.N-1)
            for j = 1:(settings.N-2)
                if iseven(i+j)
                    B[i,j+1] = 0.0
                else
                    B[i,j+1] = -4.0*nu*(2*(i+1)+1)*min(i-1,j)*(min(i-1,j)+1)/2
                end
            end
        end

        # setup basis functions for velocity profile
        Nalpha = settings.N-2;
        L0 = collectPl(-1.0, lmax = Nalpha);
        nq = 4*N;
        xi,w = gausslegendre(nq);

        phi0 = zeros(nq);
        phi0W = zeros(nq);
        phi = zeros(Nalpha,nq);
        phiW = zeros(Nalpha,nq);

        for k = 1:length(xi)
            tmp = collectPl(xi[k], lmax = (Nalpha))./L0;
            phi0[k] = tmp[0];
            phi0W[k] = tmp[0]*w[k]/2;
            phi[:,k] .= tmp[1:end];
            phiW[:,k] .= tmp[1:end]*w[k].*(2*(1:Nalpha) .+ 1)/2;
        end
        uncertParam = "0";
        sample = [0.2];

        new(x,gridSize,gridWidth,settings,outRhsSW,outRhsCorr,outRhs,gamma,A,B,C,settings.r,phi0,phi0W,phi,phiW,uncertParam,sample);
    end
end

function SetupIC(obj::solver,N::Int)
    u = zeros(obj.settings.NCells,N); # Nx interfaces, means we have Nx - 1 spatial cells
    x = obj.settings.xMid
    u[:,1] = IC(obj.settings,obj.sample,obj.uncertParam);

    N = obj.settings.N-2;
    alpha = zeros(obj.settings.NCells,N);
    L0 = collectPl(-1.0, lmax = N+1);
    nq = size(obj.phiW,2)
    xi,w = gausslegendre(nq);
    h = 0.5*(xi .+ 1.0);#[end:-1:1];
    uMax = 1.0;

    if obj.settings.problem == "sqrt"
        vProfile = uMax*(h).^(1/10)
    elseif obj.settings.problem == "TwoLayer"
        uMax = 0.15;
        vProfile = uMax*ones(size(h));
        idx = findall(h .<= 0.2);
        vProfile[idx] .= 0.1;
    else
        vProfile = zeros(size(h));
    end

    if obj.settings.problem == "KowalskiTorrihon"
        Nalpha = N
        Null = zeros(1,Nalpha);
        One  = ones(size(h));
        alpha1 = -0.25;
        alphaN = -alpha1;
        um = 0.25;
        N = obj.settings.N;
        Nalpha = N -2;
        for j = 1:obj.settings.NCells
            alpha[j,:] = Null*u[j,1];
            alpha[j,1] = alpha1*u[j,1];
            alpha[j,Nalpha] = alphaN*u[j,1];
            u[j,2] = obj.phi0W'*One*um*u[j,1];
        end
    else
        for j = 1:obj.settings.NCells
            alpha[j,:] = obj.phiW*vProfile*u[j,1];
            u[j,2] = obj.phi0W'*vProfile*u[j,1];
        end
    end
    return u,alpha
end


function SetupICFull(obj::solver,N::Int)
    u = zeros(obj.settings.NCells,N); # Nx interfaces, means we have Nx - 1 spatial cells
    x = obj.settings.xMid
    u[:,1] = IC(obj.settings,x);

    N = obj.settings.N-2;
    alpha = zeros(obj.settings.NCells,N);
    L0 = collectPl(-1.0, lmax = N+1);
    nq = size(obj.phiW,2)
    xi,w = gausslegendre(nq);
    h = 0.5*(xi .+ 1.0);#[end:-1:1];
    uMax = 1.0;

    if obj.settings.problem == "sqrt"
        vProfile = uMax*(h).^(1/4)
    elseif obj.settings.problem == "TwoLayer"
        uMax = 0.15;
        vProfile = uMax*ones(size(h));
        idx = findall(h .<= 0.2);
        vProfile[idx] .= 0.1;
    else
        vProfile = zeros(size(h));
    end

    if obj.settings.problem == "KowalskiTorrihon"
        Nalpha = N
        Null = zeros(1,Nalpha);
        One  = ones(size(h));
        alpha1 = -0.25;
        alphaN = -alpha1;
        um = 0.25;
        N = obj.settings.N;
        Nalpha = N -2;
        for j = 1:obj.settings.NCells
            alpha[j,:] = Null*u[j,1];
            alpha[j,1] = alpha1*u[j,1];
            alpha[j,Nalpha] = alphaN*u[j,1];
            u[j,2] = obj.phi0W'*One*um*u[j,1];
        end
    else
        for j = 1:obj.settings.NCells
            alpha[j,:] = obj.phiW*vProfile*u[j,1];
            u[j,2] = obj.phi0W'*vProfile*u[j,1];
        end
    end
    return [u alpha]
end

function Rhs(obj::solver,u::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    AMinus = FluxMatrix(obj,0.5*(u[1,:]+u[2,:]));
    APlus = zeros(size(AMinus))
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        APlus .= FluxMatrix(obj,0.5*(u[j,:]+u[j+1,:]));

        obj.outRhs[j,:] = -APlus*(u[j+1,:]-u[j,:])/dx/2 - AMinus*(u[j,:]-u[j-1,:])/dx/2

        obj.outRhs[j,:] .+= (u[j+1,:]-2*u[j,:]+u[j-1,:])/dt/2;

        AMinus .= APlus
    end

    return obj.outRhs;
end

function fluxSW(obj::solver,h::Float64,hu::Float64,hα::Float64)
    α = hα / h;
    return [hu; h*hu + h*hα / 3 + obj.settings.g / 2 * h^2];
end

# right hand side for classical shallow water part with alpha1 correction
function RhsSWCons(obj::solver,u::Array{Float64,2},alphaH1::Array{Float64,1},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells

        fPlus = 0.5 * (fluxSW(obj,u[j,1],u[j,2],alphaH1[j])+fluxSW(obj,u[j+1,1],u[j+1,2],alphaH1[j+1])) .- dx/dt/2 * ([u[j+1,1]-u[j,1],u[j+1,2] - u[j,2]])
        fMinus = 0.5 * (fluxSW(obj,u[j-1,1],u[j-1,2],alphaH1[j-1])+fluxSW(obj,u[j,1],u[j,2],alphaH1[j])) .- dx/dt/2 * ([u[j,1]-u[j-1,1],u[j,2] - u[j-1,2]])

        obj.outRhsSW[j,:] = -1/dx * (fPlus - fMinus)

    end

    return obj.outRhsSW;
end

# right hand side for classical shallow water part with alpha1 correction
function RhsSW(obj::solver,u::Array{Float64,2},alphaH1::Array{Float64,1},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    alpha1 = alphaH1./u[:,1];
    AMinus,BMinus = FluxMatrixSW(obj,0.5*(u[1,:]+u[2,:]),0.5*(alpha1[1]+alpha1[2]));
    APlus = zeros(size(AMinus))
    BPlus = zeros(size(BMinus))
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        APlus,BPlus = FluxMatrixSW(obj,0.5*(u[j,:]+u[j+1,:]),0.5*(alpha1[j]+alpha1[j+1]));

        obj.outRhsSW[j,:] = -APlus*(u[j+1,:]-u[j,:])/dx/2 - AMinus*(u[j,:]-u[j-1,:])/dx/2
        obj.outRhsSW[j,:] += -BPlus*(alphaH1[j+1]-alphaH1[j])/dx/2 - BMinus*(alphaH1[j]-alphaH1[j-1])/dx/2

        obj.outRhsSW[j,:] .+= (u[j+1,:]-2*u[j,:]+u[j-1,:])/dt/2;

        AMinus .= APlus
        BMinus .= BPlus
    end

    return obj.outRhsSW;
end

function Rhs(obj::solver,u::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    AMinus = FluxMatrix(obj,0.5*(u[1,:]+u[2,:]));
    APlus = zeros(size(AMinus))
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        APlus .= FluxMatrix(obj,0.5*(u[j,:]+u[j+1,:]));

        obj.outRhs[j,:] = -APlus*(u[j+1,:]-u[j,:])/dx/2 - AMinus*(u[j,:]-u[j-1,:])/dx/2

        obj.outRhs[j,:] .+= (u[j+1,:]-2*u[j,:]+u[j-1,:])/dt/2;

        AMinus .= APlus
    end

    return obj.outRhs;
end

function RhsCorrection(obj::solver,u::Array{Float64,2},alphaH::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    alpha1 = alphaH[:,1]./u[:,1];
    AMinus, BMinus = FluxMatrixCorrection(obj,0.5*(u[1,:]+u[2,:]),0.5*(alpha1[1]+alpha1[2]));
    APlus = zeros(size(AMinus))
    BPlus = zeros(size(BMinus))
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        APlus,BPlus = FluxMatrixCorrection(obj,0.5*(u[j,:]+u[j+1,:]),0.5*(alpha1[j]+alpha1[j+1]));

        obj.outRhsCorr[j,:] = -APlus*(alphaH[j+1,:]-alphaH[j,:])/dx/2 - AMinus*(alphaH[j,:]-alphaH[j-1,:])/dx/2
        obj.outRhsCorr[j,:] += -BPlus*(u[j+1,:]-u[j,:])/dx/2 - BMinus*(u[j,:]-u[j-1,:])/dx/2

        obj.outRhsCorr[j,:] .+= (alphaH[j+1,:]-2*alphaH[j,:]+alphaH[j-1,:])/dt/2;

        AMinus .= APlus
        BMinus .= BPlus
    end

    return obj.outRhsCorr;
end

function RhsCorrectionK(obj::solver,u::Array{Float64,2},K::Array{Float64,2},W::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    N = obj.settings.N

    alpha1 = K*W[1,:]./u[:,1];

    WAW =  W'*obj.A*W;

    alphaMinus = 0.5*(alpha1[1]+alpha1[2])
    uMminus = 0.5*(u[1,2]+u[2,2])/(0.5*(u[1,1]+u[2,1]));
    WAminusW = WAW *alphaMinus + I*uMminus;
    WAplusW = zeros(size(WAminusW));
    outRhsCorr = zeros(size(K));

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells

        alphaPlus = 0.5*(alpha1[j]+alpha1[j+1])
        uMplus = 0.5*(u[j,2]+u[j+1,2])/(0.5*(u[j,1]+u[j+1,1]));

        WAplusW .= WAW *alphaPlus + I*uMplus;

        outRhsCorr[j,:] = -WAplusW*(K[j+1,:]-K[j,:])/dx/2 - WAminusW*(K[j,:]-K[j-1,:])/dx/2

        outRhsCorr[j,:] .+= ((u[j+1,1]-u[j,1])*uMplus*alphaPlus/dx + (u[j,1]-u[j-1,1])*uMminus*alphaMinus/dx)*W[1,:]
        outRhsCorr[j,:] .+= (-(u[j+1,2]-u[j,2])*alphaPlus/dx - (u[j,2]-u[j-1,2])*alphaMinus/dx)*W[1,:]
        outRhsCorr[j,:] .+= (1/3*(u[j+1,1]-u[j,1])*alphaPlus.^2/dx + 1/3*(u[j,1]-u[j-1,1])*alphaMinus^2/dx)*W[2,:]

        outRhsCorr[j,:] .+= (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2;

        WAminusW .= WAplusW
        alphaMinus = alphaPlus
        uMminus = uMplus;
    end

    return outRhsCorr;
end

function RhsCorrectionK(obj::solver,u::Array{Float64,2},K::Array{Float64,2},W::Array{Float64,2},WAW::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    N = obj.settings.N

    alpha1 = K*W[1,:]./u[:,1];
    alphaMinus = 0.5*(alpha1[1]+alpha1[2])
    uMminus = 0.5*(u[1,2]+u[2,2])/(0.5*(u[1,1]+u[2,1]));
    WAminusW = WAW *alphaMinus + I*uMminus;
    WAplusW = zeros(size(WAminusW));
    outRhsCorr = zeros(size(K));

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells

        alphaPlus = 0.5*(alpha1[j]+alpha1[j+1])
        uMplus = 0.5*(u[j,2]+u[j+1,2])/(0.5*(u[j,1]+u[j+1,1]));

        WAplusW .= WAW *alphaPlus + I*uMplus;

        outRhsCorr[j,:] = -WAplusW*(K[j+1,:]-K[j,:])/dx/2 - WAminusW*(K[j,:]-K[j-1,:])/dx/2

        outRhsCorr[j,:] .+= ((u[j+1,1]-u[j,1])*uMplus*alphaPlus/dx + (u[j,1]-u[j-1,1])*uMminus*alphaMinus/dx)*W[1,:]
        outRhsCorr[j,:] .+= (-(u[j+1,2]-u[j,2])*alphaPlus/dx - (u[j,2]-u[j-1,2])*alphaMinus/dx)*W[1,:]
        outRhsCorr[j,:] .+= (1/3*(u[j+1,1]-u[j,1])*alphaPlus.^2/dx + 1/3*(u[j,1]-u[j-1,1])*alphaMinus^2/dx)*W[2,:]

        outRhsCorr[j,:] .+= (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2;

        WAminusW .= WAplusW
        alphaMinus = alphaPlus
        uMminus = uMplus;
    end

    return outRhsCorr;
end

function RhsCorrectionL(obj::solver,u::Array{Float64,2},X::Array{Float64,2},L::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;
    N = obj.settings.N

    alpha1 = X*L[1,:]./u[:,1];

    alphaPlus = 0.5*(alpha1[1]+alpha1[2])
    uMplus = 0.5*(u[1,2]+u[2,2])/(0.5*(u[1,1]+u[2,1]));
    outRhsCorr = zeros(size(L));
    alphaMinus = alphaPlus;
    uMminus = uMplus;
    r = size(X,2);

    # compute spatial flux matrix
    XX1 = zeros(r,r);
    XX2 = zeros(r,r);
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells

        alphaPlus = 0.5*(alpha1[j]+alpha1[j+1])
        uMplus = 0.5*(u[j,2]+u[j+1,2])/(0.5*(u[j,1]+u[j+1,1]));

        XX1 .+= -alphaPlus*X[j,:]*(X[j+1,:]-X[j,:])'/dx/2-alphaMinus*X[j,:]*(X[j,:]-X[j-1,:])'/dx/2
        XX2 .+= -uMminus*X[j,:]*(X[j,:]-X[j-1,:])'/dx/2 -uMplus*X[j,:]*(X[j+1,:]-X[j,:])'/dx/2 + X[j,:]*(X[j+1,:]-2*X[j,:]+X[j-1,:])'/dt/2

        outRhsCorr[1,:] .+= X[j,:]*(u[j+1,1]-u[j,1])*uMplus*alphaPlus/dx + X[j,:]*(u[j,1]-u[j-1,1])*uMminus*alphaMinus/dx
        outRhsCorr[1,:] .+= -X[j,:]*(u[j+1,2]-u[j,2])*alphaPlus/dx - X[j,:]*(u[j,2]-u[j-1,2])*alphaMinus/dx
        outRhsCorr[2,:] .+= 1/3*X[j,:]*(u[j+1,1]-u[j,1])*alphaPlus.^2/dx + 1/3*X[j,:]*(u[j,1]-u[j-1,1])*alphaMinus^2/dx

        alphaMinus = alphaPlus
        uMminus = uMplus;
    end

    outRhsCorr .+= obj.A*L*XX1' .+ L*XX2';

    return outRhsCorr;
end

function RhsCorrectionS(obj::solver,u::Array{Float64,2},X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},t::Float64=0.0)
    dt = obj.settings.dt;
    dx = obj.settings.dx;

    alpha1 = X*S*W[1,:]./u[:,1];

    alphaMinus = 0.5*(alpha1[1]+alpha1[2])
    uMminus = 0.5*(u[1,2]+u[2,2])/(0.5*(u[1,1]+u[2,1]));

    outRhsCorr = zeros(size(S));

    r = size(X,2);

    # compute spatial flux matrix and B part
    XX1 = zeros(r,r);
    XX2 = zeros(r,r);
    B1 = zeros(r);
    B2 = zeros(r);
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells

        alphaPlus = 0.5*(alpha1[j]+alpha1[j+1])
        uMplus = 0.5*(u[j,2]+u[j+1,2])/(0.5*(u[j,1]+u[j+1,1]));

        XX1 .+= -alphaPlus*X[j,:]*(X[j+1,:]-X[j,:])'/dx/2-alphaMinus*X[j,:]*(X[j,:]-X[j-1,:])'/dx/2;
        XX2 .+= -uMminus*X[j,:]*(X[j,:]-X[j-1,:])'/dx/2 -uMplus*X[j,:]*(X[j+1,:]-X[j,:])'/dx/2 + X[j,:]*(X[j+1,:]-2*X[j,:]+X[j-1,:])'/dt/2;

        B1 .+= (X[j,:]*(u[j+1,1]-u[j,1])*uMplus*alphaPlus/dx + X[j,:]*(u[j,1]-u[j-1,1])*uMminus*alphaMinus/dx);
        B1 .+= (-X[j,:]*(u[j+1,2]-u[j,2])*alphaPlus/dx - X[j,:]*(u[j,2]-u[j-1,2])*alphaMinus/dx);
        B2 .+= (1/3*X[j,:]*(u[j+1,1]-u[j,1])*alphaPlus.^2/dx + 1/3*X[j,:]*(u[j,1]-u[j-1,1])*alphaMinus^2/dx);

        alphaMinus = alphaPlus;
        uMminus = uMplus;
    end

    WAW = W'*obj.A'*W;

    outRhsCorr .+= XX1*S*WAW .+ XX2*S .+ B1*W[1,:]' .+ B2*W[2,:]';

    return outRhsCorr;
end

function FluxMatrix(obj::solver,u::Array{Float64,1})
    N = obj.settings.N
    A = zeros(obj.settings.N,obj.settings.N)
    h = u[1]
    uM = u[2]/h
    alpha1 = u[3]/h
    g = obj.settings.g

    A[1,2] = 1;

    A[2,1] = g*h-uM*uM-1/3*alpha1^2;
    A[2,2] =2*uM;
    A[2,3] =2/3*alpha1;

    A[3,1] = -2*uM*alpha1;
    A[3,2] =2*alpha1;
    A[3,3] =uM;
    A[3,4] = 3/5*alpha1;

    A[4,1]= -2/3*alpha1^2;
    A[4,3] = 1/3*alpha1;
    A[4,4] = uM;

    for i=4:(N-1)
        A[i+1,i] = ((i-2)/(2*(i-1)-1))*alpha1;
        A[i,i+1]= ((i)/(2*(i-1)+1))*alpha1;
        A[i+1,i+1]= uM;
    end

    return A
end

function FluxMatrixSW(obj::solver,u::Array{Float64,1},alpha1::Float64)
    A = zeros(2,2)
    B = zeros(2,1)
    h = u[1]
    uM = u[2]/h

    g = obj.settings.g

    A[1,2] = 1;

    A[2,1] = g*h-uM*uM-1/3*alpha1^2;
    A[2,2] =2*uM;
    B[2,1] =2/3*alpha1;

    return A,B
end

function FluxMatrixCorrection(obj::solver,u::Array{Float64,1},alpha1::Float64)
    N = obj.settings.N
    A = zeros(N-2,N-2)
    h = u[1]
    uM = u[2]/h

    for i=1:(N-3)
        n = i+1
        A[i,i+1] = (n+1)/(2*n+1)*alpha1;
        A[i+1,i]= (n-1)/(2*n-1)*alpha1;
    end

    B = zeros(N-2,2)

    B[1,1] = -2*uM*alpha1;
    B[1,2] = 2*alpha1;
    if N-2 > 1
        B[2,1] = -2/3*alpha1^2;
    end

    return A+I*uM, B
end

function Source(obj::solver,u::Array{Float64,2})
    h = u[1]
    N = obj.settings.N
    dt = obj.settings.dt

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        h = u[j,1]
        D = (I-dt.*obj.B./(h^2)-dt.*obj.C./h)
        u[j,2:end] .= D\u[j,2:end]
    end

    return u
end

function SourceCorrectionAlphaFull(obj::solver,u::Array{Float64,2},alpha::Array{Float64,2})
    dt = obj.settings.dt
    X,S,W = svd(alpha);
    S = diagm(S);
    r = size(S,1);
    h = u[:,1];
    nMoments = obj.settings.N-2;
    nx = obj.settings.NCells

    Xhinv2X = X'*Diagonal(1.0./(h.^2))*X;
    XhinvX = X'*Diagonal(1.0./(h))*X;
    WBW = W'*obj.B[2:end,2:end]'*W
    WCW = W'*obj.C[2:end,2:end]'*W
    umX = ((1.0./h).*u[:,2])'*X
    CW = obj.C[2:end,1]'*W;
    S0 = deepcopy(S);

    yVec = mat2vec(X'*alpha*W .+ dt.*umX'*CW);
    D = diagm(ones(r*r));
    for k = 1:r
        for i = 1:r
            for l = 1:r
                for j = 1:r
                    D[(j-1) * r + k,(i-1) * r + l] -= dt*(Xhinv2X[k,l]*WBW[i,j] + XhinvX[k,l]*WCW[i,j])
                end
            end
        end
    end
    S = vec2mat(D\yVec,r,r);

    return X*S*W';
end

function SourceCorrectionAlpha(obj::solver,u::Array{Float64,2},alpha::Array{Float64,2})
    dt = obj.settings.dt
    h = u[:,1];

    Xhinv2X = Diagonal(1.0./(h.^2));
    XhinvX = Diagonal(1.0./(h));
    WBW = obj.B[2:end,2:end]'
    WCW = obj.C[2:end,2:end]'
    umX = ((1.0./h).*u[:,2])'
    CW = obj.C[2:end,1]';
    S0 = deepcopy(alpha);

    yVec = S0 .+ dt.*umX'*CW;
    for j = 1:obj.settings.NCells # leave out ghost cells
        h = u[j,1]
        D = (I-dt.*obj.B[2:end,2:end]./(h^2)-dt.*obj.C[2:end,2:end]./h);
        alpha[j,:] = D \ yVec[j,:]#(alpha[j,:]+dt.*obj.C[2:end,1]./h.*u[j,2]);
    end

    return alpha
end

function SourceCorrectionAlphaKStep(obj::solver,u::Array{Float64,2},K::Array{Float64,2},W::Array{Float64,2})
    dt = obj.settings.dt
    WBW = W'*obj.B[2:end,2:end]*W;
    WCW = W'*obj.C[2:end,2:end]*W;
    CW = collect(obj.C[2:end,1]'*W);

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        h = u[j,1];
        D = (I-dt.*WBW./(h^2)-dt.*WCW./h);
        K[j,:] = D \ (K[j,:].+dt.*CW'./h.*u[j,2]);
    end

    return K;
end

function SourceCorrectionAlphaKStep(obj::solver,u::Array{Float64,2},K::Array{Float64,2},W::Array{Float64,2},WBW::Array{Float64,2},WCW::Array{Float64,2},CW::Array{Float64,2})
    dt = obj.settings.dt

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        h = u[j,1];

        D = (I-dt.*WBW./(h^2)-dt.*WCW./h);
        #D = W'*(I-dt.*obj.B[2:end,2:end]./(h^2)-dt.*obj.C[2:end,2:end]./h)*W;
        #K[j,:] = D \ (K[j,:].+(dt.*obj.C[2:end,1]'./h.*u[j,2]*W)')
        K[j,:] = D \ (K[j,:].+dt.*CW'./h.*u[j,2]);
    end

    return K;
end

function SourceCorrectionAlphaLStep(obj::solver,u::Array{Float64,2},X::Array{Float64,2},L::Array{Float64,2})
    dt = obj.settings.dt
    r = size(L,2);
    h = u[:,1];
    Xhinv2X = X'*Diagonal(1.0./h.^2)*X;
    XhinvX = X'*Diagonal(1.0./h)*X;
    nMoments = obj.settings.N-2;
    umX = ((1.0./h).*u[:,2])'*X
    B = obj.B[2:end,2:end];
    C = obj.C[2:end,2:end];

    yVec = 0.0*mat2vec(L + dt.*obj.C[2:end,1]*umX );
    yMat = L + dt.*obj.C[2:end,1]*umX;
    D = diagm(ones(nMoments*r));
    for k = 1:nMoments
        for j = 1:r
            yVec[(j-1) * nMoments + k] = yMat[k,j]
            for l = 1:nMoments
                for i = 1:r
                    D[(j-1) * nMoments + k,(i-1) * nMoments + l] -= dt*(Xhinv2X[j,i]*C[k,l] + XhinvX[j,i]*B[k,l])
                end
            end
        end
    end
    L = vec2mat(D\yVec,nMoments,r);

    return L
end

function SourceCorrectionAlphaSStep(obj::solver,u::Array{Float64,2},X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    dt = obj.settings.dt
    r = size(S,1);
    h = u[:,1];
    nMoments = obj.settings.N-2;
    nx = obj.settings.NCells

    Xhinv2X = X'*Diagonal(1.0./(h.^2))*X;
    XhinvX = X'*Diagonal(1.0./(h))*X;
    WBW = W'*obj.B[2:end,2:end]'*W
    WCW = W'*obj.C[2:end,2:end]'*W
    umX = ((1.0./h).*u[:,2])'*X
    CW = obj.C[2:end,1]'*W;
    S0 = deepcopy(S);

    yVec = mat2vec(S0 .+ dt.*umX'*CW);
    D = diagm(ones(r*r));
    for k = 1:r
        for i = 1:r
            for l = 1:r
                for j = 1:r
                    D[(j-1) * r + k,(i-1) * r + l] -= dt*(Xhinv2X[k,l]*WBW[i,j] + XhinvX[k,l]*WCW[i,j])
                end
            end
        end
    end
    S = vec2mat(D\yVec,r,r);

    return S;
end

function SourceCorrectionUm(obj::solver,u::Array{Float64,2},alpha::Array{Float64,2})
    dt = obj.settings.dt
    B = obj.B[1,2:end];
    C = obj.C[1,2:end];
    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        h = u[j,1];
        u[j,2] = (u[j,2] + dt.*(B'alpha[j,:])/(h^2) + dt.*(C'alpha[j,:])/h)/(1.0-dt*obj.C[1,1]/h);
    end
    return u
end

function SourceCorrectionKStep(obj::solver,u::Array{Float64,2},K::Array{Float64,2},W::Array{Float64,2})
    N = obj.settings.N
    dt = obj.settings.dt
    r = size(W,2);

    Wext = zeros(N-1,r+1);

    Wext[1,1] = 1.0;
    Wext[2:end,2:end] .= W;

    WBW = Wext'*obj.B*Wext;
    WCW = Wext'*obj.C*Wext;

    for j = 2:(obj.settings.NCells-1) # leave out ghost cells
        h = u[j,1]
        D = (I-dt.*WBW./(h^2)-dt.*WCW./h);

        y = [u[j,2];K[j,:]]
        x = D\y
        u[j,2] = x[1];
        K[j,:] = x[2:end];
    end

    XNew,STmp = qr(K);
    XNew = Matrix(XNew[:, 1:r]);

    return u,XNew
end

function SourceCorrectionLStep(obj::solver,u::Array{Float64,2},X::Array{Float64,2},L::Array{Float64,2})
    h = u[:,1]
    N = obj.settings.N
    dt = obj.settings.dt
    r = size(X,2);

    Xhinv2X = X'*diagm(1.0./h.^2)*X;
    XhinvX = X'*diagm(1.0./h)*X;

    D = zeros((N-1)*r,(N-1)*r)

    y = [(X'*u[:,2])';L]

    yVec = mat2vec(Matrix(y));
    for k = 1:(N-1)
        for i = 1:r
            for l = 1:(N-1)
                for j = 1:r
                    D[(i-1) * (N-1) + k,(j-1) * (N-1) + l] -= dt*(Xhinv2X[i,j]*obj.B[k,l] + XhinvX[i,j]*obj.C[k,l])
                    if i == j && k == l
                        D[(i-1) * (N-1) + k,(j-1) * (N-1) + l] += 1.0;
                    end
                end
            end
        end
    end
    x = vec2mat(D\yVec,N-1,r);
    u[:,2] = X*x[1,:];
    L = x[2:end,:];

    WNew,STmp = qr(L);
    WNew = Matrix(WNew[:, 1:r]);

    return u,WNew
end

function SourceCorrectionSStep(obj::solver,u::Array{Float64,2},X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    h = u[:,1]
    N = obj.settings.N
    dt = obj.settings.dt
    nx = obj.settings.NCells;
    r = size(X,2);

    Wext = zeros(N-1,r+1);
    Xext = zeros(nx,r+1);

    Wext[1,1] = 1.0;
    Wext[2:end,2:end] .= W;
    Xext = [u[:,2]/norm(u[:,2]) X];

    WBW = Wext'*obj.B*Wext;
    WCW = Wext'*obj.C*Wext;

    Xhinv2X = Xext'*diagm(1.0./h.^2)*Xext;
    XhinvX = Xext'*diagm(1.0./h)*Xext;

    D = zeros((r+1)^2,(r+1)^2)

    Sext = zeros(r+1,r+1);
    Sext[1,1] = norm(u[:,2]);
    Sext[2:end,2:end] .= S;

    SVec = mat2vec(Matrix(Sext));
    for k = 1:(r+1)
        for i = 1:(r+1)
            for l = 1:(r+1)
                for j = 1:(r+1)
                    D[(k-1) * (r+1) + i,(l-1) * (r+1) + j] -= dt*(Xhinv2X[i,j]*WBW[k,l] + XhinvX[i,j]*WCW[k,l])
                    if i == j && k == l
                        D[(k-1) * (r+1) + i,(l-1) * (r+1) + j] += 1.0;
                    end
                end
            end
        end
    end
    x = vec2mat(D\SVec,r+1,r+1);
    u[:,2] = Xext*x[:,1];
    S .= x[2:end,2:end];

    return u,S
end

function mat2vec(M::Array{Float64,2})
    m = size(M,1);
    n = size(M,2);
    y = zeros(n*m);
    for k = 1:m
        for i = 1:n
            y[(i-1) * m + k] = M[k,i];
        end
    end
    return y;
end

function vec2mat(v::Array{Float64,1},m::Int,n::Int)
    y = zeros(m,n);
    for k = 1:m
        for i = 1:n
            y[k,i] = v[(i-1) * m + k];
        end
    end
    return y;
end

function Solve(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    # Set up initial condition
    u = SetupICFull(obj,2);

    BC = "periodic"
    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    Nt = Integer(round(Tend/dt));

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt

        # Update streaming in time by dt
        yU = Rhs(obj,u);
        u .= u .+ dt*yU;

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        # implicit source update
        u .= Source(obj,u)

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        t = t+dt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, u;

end

function SolveDLRAnaive(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = s.r;

    # Set up initial condition
    u = SetupICFull(obj,2);

    BC = "periodic"
    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N,r);

    Nt = Integer(round(Tend/dt));

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt

        yU = Rhs(obj,X*S*W');

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*yU*W;

        XNew,_ = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*yU'*X;

        WNew,_ = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;

        W = WNew;
        X = XNew;

        ################# S-step #################
        S = MUp*S*(NUp');
        u = X*S*W'
        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        yU = Rhs(obj,u);

        S = S + dt*X'*yU*W;

        u = X*S*W'

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        # implicit source update
        u .= Source(obj,u)

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        X,S,W = svd(u)

        S = diagm(S);

        t = t+dt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, u;

end

function SolveSW(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    BC = "periodic"
    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end
    uNew = deepcopy(u);

    Nt = Integer(round(Tend/dt));



    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt
        # Update streaming in time by dt
        yU = RhsSW(obj,u,zeros(obj.settings.NCells));
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        uNew = SourceCorrectionUm(obj,uNew,zeros(size(alpha)));

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u .= uNew;

        t = t+dt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, u,alpha;

end

function SolveCorrectionFormulation(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    BC = "periodic"
    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = deepcopy(u);

    Nt = Integer(round(Tend/dt));

    # time loop
    for n = 1:Nt

        # Update streaming in time by dt
        yU = RhsSW(obj,u,alpha[:,1]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,alpha);

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        yU = RhsCorrection(obj,u,alpha);
        alpha .= alpha .+ dt*yU;

        if BC == "periodic"
            alpha[1,:] = alpha[end-1,:]
            alpha[end,:] = alpha[2,:]
        end

        # implicit source update
        #uNew,alpha = SourceCorrectionOld(obj,uNew,alpha)
        alpha = SourceCorrectionAlpha(obj,uNew,alpha);
        #alpha = SourceCorrectionAlphaFull(obj,uNew,alpha);

        if BC == "periodic"
            alpha[1,:] = alpha[end-1,:]
            alpha[end,:] = alpha[2,:]
        end

        t = t+dt;
    end
    rVec = min(size(u)) .*ones(Nt);
    # return end time and solution
    return t, u,alpha,rVec;

end

function FullProblem(obj::solver, sample::Array{Float64,1})
    obj.sample = sample;
    _,u,g,rVec = SolveCorrectionFormulation(obj);
    return u,rVec;
end

function SolveCorrectionFormulationUnconventional(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;

    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);

    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = zeros(size(u));

    # Low-rank approx of init data:
    X,S,W = svd(alpha);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));

    # time loop
    for n = 1:Nt

        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,X*S*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,X*S*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*RhsCorrectionK(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*RhsCorrectionL(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;
        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp')
        #S = S + dt*X'*RhsCorrection(obj,u,X*S*W')*W;

        S = S + dt*RhsCorrectionS(obj,u,X,S,W);

        #####################################
        #   Update friction in time by dt   #
        #####################################

        ################# K-step #################

        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K .= SourceCorrectionAlphaKStep(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L .= SourceCorrectionAlphaLStep(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= SourceCorrectionAlphaSStep(obj,u,X,S,W);

        t = t+dt;
    end
    rVec = s.r .*ones(Nt)
    # return end time and solution
    return t, u,X*S*W',rVec;

end

function fixedrankBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample;
    _, u, g, rVec = SolveCorrectionFormulationUnconventional(obj);
    return u, rVec;
end

function SolveCorrectionFormulationParallel(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;

    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);

    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = zeros(size(u));

    # Low-rank approx of init data:
    X,S,W = svd(alpha);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));
    ranks = zeros(Nt,2);

    # time loop
    for n = 1:Nt

        ranks[n,1] = t;
        ranks[n,2] = r;

        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,X*S*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,X*S*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*RhsCorrectionK(obj,u,K,W);

        if BC == "periodic"
            X[1,:] = X[end-1,:]
            X[end,:] = X[2,:]
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        X₁,_ = qr([X K]);
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(r+1):(2*r)];

        ################# L-step #################
        L = W*S';
        L = L + dt*RhsCorrectionL(obj,u,X,L);

        W₁,_ = qr([W L]);
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(r+1):(2*r)];

        ################## S-step ##################
        S = S + dt*RhsCorrectionS(obj,u,X,S,W);

        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = S;
        SNew[(r+1):end,1:r] = tildeX₁'*K;
        SNew[1:r,(r+1):(2 * r)] = L' * tildeW₁;

        # truncate
        X, S, W = truncate!(obj,[X tildeX₁],SNew,[W tildeW₁]);

        # update rank
        r = size(S,1);

        #####################################
        #   Update friction in time by dt   #
        #####################################

        ################# K-step #################

        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K .= SourceCorrectionAlphaKStep(obj,u,K,W);

        if BC == "periodic"
            X[1,:] = X[end-1,:]
            X[end,:] = X[2,:]
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        X₁,_ = qr([X K]);
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(r+1):(2*r)];

        ################# L-step #################
        L = W*S';
        L .= SourceCorrectionAlphaLStep(obj,u,X,L);

        W₁,_ = qr([W L]);
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(r+1):(2*r)];

        ################## S-step ##################

        S .= SourceCorrectionAlphaSStep(obj,u,X,S,W);

        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = S;
        SNew[(r+1):end,1:r] = tildeX₁'*K;
        SNew[1:r,(r+1):(2 * r)] = L' * tildeW₁;

        # truncate
        X, S, W = truncate!(obj,[X tildeX₁],SNew,[W tildeW₁]);

        # update rank
        r = size(S,1);

        t = t+dt;
    end

    # return end time and solution
    return t, u,X*S*W', ranks;

end

function parBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample;
    _, u, g, rVec = SolveCorrectionFormulationParallel(obj);
    return u, rVec;
end

function SolveCorrectionFormulationRankAdaptBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;

    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);

    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = zeros(size(u));

    # Low-rank approx of init data:
    X,S,W = svd(alpha);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));
    ranks = zeros(Nt,2);

    # time loop
    for n = 1:Nt

        ranks[n,1] = t;
        ranks[n,2] = r;
        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,X*S*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,X*S*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*RhsCorrectionK(obj,u,K,W);

        if BC == "periodic"
            X[1,:] = X[end-1,:]
            X[end,:] = X[2,:]
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        XNew,_ = qr([K X]);
        XNew = Matrix(XNew[:, 1:2*r]);

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*RhsCorrectionL(obj,u,X,L);

        WNew,_ = qr([L W]);
        WNew = Matrix(WNew[:, 1:2*r]);

        NUp = WNew' * W;

        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp')
        S = S + dt*RhsCorrectionS(obj,u,X,S,W);

        # truncate
        X, S, W = truncate!(obj,X,S,W);

        # update rank
        r = size(S,1);

        #####################################
        #   Update friction in time by dt   #
        #####################################

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end
        K .= SourceCorrectionAlphaKStep(obj,u,K,W);

        if BC == "periodic"
            X[1,:] = X[end-1,:]
            X[end,:] = X[2,:]
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        XNew,STmp = qr([K X]);
        XNew = Matrix(XNew[:, 1:2*r]);

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L .= SourceCorrectionAlphaLStep(obj,u,X,L);

        WNew,STmp = qr([L W]);
        WNew = Matrix(WNew[:, 1:2*r]);

        NUp = WNew' * W;

        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp');

        S .= SourceCorrectionAlphaSStep(obj,u,X,S,W);

        # truncate
        X, S, W = truncate!(obj,X,S,W);

        # update rank
        r = size(S,1);

        t = t+dt;
    end

    # return end time and solution
    return t, u, X*S*W',ranks;
end

function augBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample;
    _, u, g, rVec = SolveCorrectionFormulationRankAdaptBUG(obj);
    return u, rVec;
end

function truncate!(obj::solver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    ϑ = obj.settings.ϑ*norm(D)^obj.settings.ϑIndex;
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < ϑ
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
    return X*U[:, 1:rmax], diagm(D[1:rmax]), W*V[:, 1:rmax];
end

function SolveCorrectionFormulationUnconventional_hist(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;
    Nalpha = N -2;
    Nx = obj.settings.Nx
    Ncells = obj.settings.NCells
    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    uNew = zeros(size(u));
    NCells = obj.settings.NCells;
    Nt = Integer(round(Tend/dt));
    uhist = zeros(NCells,2,Nt+1);
    thist = zeros(Nt+1);
    alpha_hist = zeros(NCells,Nalpha,Nt+1)


    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = zeros(size(u));

    # Low-rank approx of init data:
    X,S,W = svd(alpha);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));

    # time loop
    for n = 1:Nt
        uhist[:,1,n]=u[:,1]
        uhist[:,2,n]=u[:,2]
        alpha_hist[:,:,n] = X*S*W'
        thist[n] = t
        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,X*S*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,X*S*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*RhsCorrectionK(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*RhsCorrectionL(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;
        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp')
        #S = S + dt*X'*RhsCorrection(obj,u,X*S*W')*W;

        S = S + dt*RhsCorrectionS(obj,u,X,S,W);

        #####################################
        #   Update friction in time by dt   #
        #####################################

        ################# K-step #################

        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K .= SourceCorrectionAlphaKStep(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L .= SourceCorrectionAlphaLStep(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= SourceCorrectionAlphaSStep(obj,u,X,S,W);

        t = t+dt;
        next!(prog) # update progress bar
    end

    uhist[:,1,Nt+1]=u[:,1]
    uhist[:,2,Nt+1]=u[:,2]
    alpha_hist[:,:,Nt+1] = K*W'
    thist[Nt+1]=t
    # return end time and solution
    return thist, uhist,alpha_hist;

end

function SolveSW_hist(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    Nx = obj.settings.NCells
    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    uNew = deepcopy(u);

    Nt = Integer(round(Tend/dt));
    uhist = zeros(Nx,2,Nt)
    BC = "periodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt

        # Update streaming in time by dt
        yU = RhsSW(obj,u,zeros(obj.settings.NCells));
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        uNew = SourceCorrectionUm(obj,uNew,zeros(size(alpha)));

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u .= uNew;

        t = t+dt;
        uhist[:,1,n] = u[:,1]
        uhist[:,2,n] = u[:,2]
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, uhist;

end

function SolveCorrectionFormulation_hist(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    uNew = deepcopy(u);
    NCells = obj.settings.NCells
    Nt = Integer(round(Tend/dt));
    uhist = zeros(NCells,2,Nt+1)
    Nalpha = size(alpha)
    alpha_hist = zeros(NCells,Nalpha[2],Nt+1)

    BC = "periodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt
        uhist[:,1,n]=u[:,1]
        uhist[:,2,n]=u[:,2]
        alpha_hist[:,:,n] = alpha
        # Update streaming in time by dt
        yU = RhsSW(obj,u,alpha[:,1]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,alpha);

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        yU = RhsCorrection(obj,u,alpha);
        alpha .= alpha .+ dt*yU;

        if BC == "periodic"
            alpha[1,:] = alpha[end-1,:]
            alpha[end,:] = alpha[2,:]
        end

        # implicit source update
        #uNew,alpha = SourceCorrectionOld(obj,uNew,alpha)
        alpha = SourceCorrectionAlpha(obj,uNew,alpha);
        #alpha = SourceCorrectionAlphaFull(obj,uNew,alpha);

        if BC == "periodic"
            alpha[1,:] = alpha[end-1,:]
            alpha[end,:] = alpha[2,:]
        end

        t = t+dt;
        next!(prog) # update progress bar
    end
    uhist[:,1,Nt+1]=u[:,1]
    uhist[:,2,Nt+1]=u[:,2]
    alpha_hist[:,:,Nt+1] = alpha
    # return end time and solution
    return t, uhist,alpha_hist;

end

function SolveDLRAnaive_hist(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = s.r;

    # Set up initial condition
    u = SetupICFull(obj,2);

    BC = "periodic"
    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N,r);

    Nt = Integer(round(Tend/dt));
    uhist = zeros(obj.settings.NCells,2,Nt+1)
    Nalpha = obj.settings.N-2
    alpha_hist = zeros(obj.settings.NCells,Nalpha,Nt+1)

    Nt = Integer(round(Tend/dt));

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt
        u = X*S*W';

        uhist[:,1,n]=u[:,1]
        uhist[:,2,n]=u[:,2]
        alpha_hist[:,:,n] = u[:,3:end]

        yU = Rhs(obj,u);

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*yU*W;

        XNew,_ = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*yU'*X;

        WNew,_ = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;

        W = WNew;
        X = XNew;

        ################# S-step #################
        S = MUp*S*(NUp');
        u = X*S*W'
        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        yU = Rhs(obj,u);

        S = S + dt*X'*yU*W;

        u = X*S*W'

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        # implicit source update
        u .= Source(obj,u)

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        X,S,W = svd(u)

        S = diagm(S);

        t = t+dt;
        next!(prog) # update progress bar
    end

    uhist[:,1,Nt+1]=u[:,1]
    uhist[:,2,Nt+1]=u[:,2]
    alpha_hist[:,:,Nt+1] = u[:,3:end]
    # return end time and solution
    return t, uhist,alpha_hist;

end

function SolveCorrectionFormulationUnconventional_hist(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;

    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);

    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
        u[1,:] = u[end-1,:]
        u[end,:] = u[2,:]
    end

    uNew = zeros(size(u));

    # Low-rank approx of init data:
    X,S,W = svd(alpha);

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    K = zeros(obj.settings.NCells,r);
    L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));

    uhist = zeros(obj.settings.NCells,2,Nt+1)
    Nalpha = obj.settings.N-2
    alpha_hist = zeros(obj.settings.NCells,Nalpha,Nt+1)

    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt

        uhist[:,1,n]=u[:,1]
        uhist[:,2,n]=u[:,2]
        alpha_hist[:,:,n] = X*S*W'

        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,X*S*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,X*S*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################
        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K = K + dt*RhsCorrectionK(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L = L + dt*RhsCorrectionL(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;
        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp')
        S = S + dt*RhsCorrectionS(obj,u,X,S,W);

        #####################################
        #   Update friction in time by dt   #
        #####################################

        ################# K-step #################

        K = X*S;

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K .= SourceCorrectionAlphaKStep(obj,u,K,W);

        XNew,STmp = qr(K);
        XNew = Matrix(XNew[:, 1:r]);

        if BC == "periodic"
            XNew[1,:] = XNew[end-1,:]
            XNew[end,:] = XNew[2,:]
        end

        MUp = XNew' * X;

        ################# L-step #################
        L = W*S';
        L .= SourceCorrectionAlphaLStep(obj,u,X,L);

        WNew,STmp = qr(L);
        WNew = Matrix(WNew[:, 1:r]);

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= SourceCorrectionAlphaSStep(obj,u,X,S,W);

        t = t+dt;
        next!(prog) # update progress bar
    end

    uhist[:,1,Nt+1]=u[:,1]
    uhist[:,2,Nt+1]=u[:,2]
    alpha_hist[:,:,Nt+1] = X*S*W'
    # return end time and solution
    return t, uhist,alpha_hist;

end


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MOR unconventional
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SolveCorrectionFormulationUnconventional_MOR(obj::solver, W)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    N = obj.settings.N;
    Nalpha = N -2;
    Nx = obj.settings.Nx
    Ncells = obj.settings.NCells
    r = obj.settings.r
    # Set up initial condition
    u,alpha = SetupIC(obj,2);
    uNew = zeros(size(u));
    NCells = obj.settings.NCells;
    Nt = Integer(round(Tend/dt));
    uhist = zeros(NCells,2,Nt+1);
    alpha_hist = zeros(NCells,Nalpha,Nt+1)

    # Low-rank approx of init data:
    #X,S,W = svd(alpha);
    # rank-r truncation:
    #X = X[:,1:r];
    W = W[:,1:r];
    #S = Diagonal(S);
    #S = S[1:r, 1:r];

    K = zeros(Ncells,r);
    K = reshape(alpha,Ncells,Nalpha)*W;
    #L = zeros(obj.settings.N-2,r);

    Nt = Integer(round(Tend/dt));

    BC = "periodic"#"nonperiodic"

    if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
    end
    # precompute matrices
    WAW = W'*obj.A*W;
    WBW = W'*obj.B[2:end,2:end]*W;
    WCW = W'*obj.C[2:end,2:end]*W;
    CW = collect(obj.C[2:end,1]'*W);
    # time loop
    prog = Progress(Nt,1)
    for n = 1:Nt
        uhist[:,1,n]=u[:,1]
        uhist[:,2,n]=u[:,2]
        alpha_hist[:,:,n] = K*W'
        ####################################
        #  Update streaming in time by dt  #
        ####################################
        yU = RhsSW(obj,u,K*W[1,:]);
        uNew .= u .+ dt*yU;

        if BC == "periodic"
            uNew[1,:] = uNew[end-1,:]
            uNew[end,:] = uNew[2,:]
        end

        u = SourceCorrectionUm(obj,uNew,K*W')

        if BC == "periodic"
            u[1,:] = u[end-1,:]
            u[end,:] = u[2,:]
        end

        ################# K-step #################

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end
        K = K + dt*RhsCorrectionK(obj,u,K,W,WAW);

        if BC == "periodic"
            K[1,:] = K[end-1,:]
            K[end,:] = K[2,:]
        end

        K .= SourceCorrectionAlphaKStep(obj,u,K,W,WBW,WCW,CW);



        t = t+dt;
        next!(prog) # update progress bar
    end

    uhist[:,1,Nt+1]=u[:,1]
    uhist[:,2,Nt+1]=u[:,2]
    alpha_hist[:,:,Nt+1] = K*W'
    # return end time and solution
    return t, uhist,alpha_hist;

end
