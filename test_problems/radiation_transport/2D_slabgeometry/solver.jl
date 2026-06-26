__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum

include("PNSystem.jl")

mutable struct solver
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};

    gridSize::Array{Float64,1}; # Required for setting QoI in MLMC
    gridWidth::Array{Float64,1}; # Required for computing norms in MLMC

    # Solver settings
    settings::settings;

    # preallocate memory for performance
    outRhs::Array{Float64,3};
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # Roe matrix
    AbsAx::SparseMatrixCSC{Float64, Int64};
    AbsAz::SparseMatrixCSC{Float64, Int64};
    # normalized Legendre Polynomials
    P::Array{Float64,2};
    # quadrature points
    mu::Array{Float64,1};
    w::Array{Float64,1};

    # functionalities of the PN system
    pn::PNSystem;

    L1x::SparseMatrixCSC{Float64, Int64};
    L1y::SparseMatrixCSC{Float64, Int64};
    L2x::SparseMatrixCSC{Float64, Int64};
    L2y::SparseMatrixCSC{Float64, Int64};

    # Paramenters related to UQ
    uncertParam::String;
    sample::Array{Float64,1};

    # constructor
    function solver(settings)
        x = settings.x;
        y = settings.y;

        # setup flux matrix
        gamma = zeros(settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        # A = zeros(settings.nPN,settings.nPN);
        #     # setup flux matrix (alternative analytic computation)
        # for i = 1:(settings.nPN-1)
        #     n = i-1;
        #     A[i,i+1] = (n+1)/(2*n+1)*sqrt(gamma[i+1])/sqrt(gamma[i]);
        # end

        # for i = 2:settings.nPN
        #     n = i-1;
        #     A[i,i-1] = n/(2*n+1)*sqrt(gamma[i-1])/sqrt(gamma[i]);
        # end

        # construct PN system matrices
        # pn = PNSystem(settings)
        # SetupSystemMatrices(pn);

        pn = PNSystem(settings,Float64)
        Ax,_,Az = SetupSystemMatrices(pn);
        SetupSystemMatricesSparse(pn);

        outRhs = zeros(settings.NCellsX,settings.NCellsY,pn.nTotalEntries);

        S = eigvals(Matrix(pn.Ax))
        V = eigvecs(Matrix(pn.Ax))
        AbsAx = V*abs.(Diagonal(S))*inv(V)

        idx = findall(abs.(AbsAx) .> 1e-10)
        Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = AbsAx[idx];
        AbsAx = sparse(Ix,Jx,Float64.(vals),pn.nTotalEntries,pn.nTotalEntries);

        S = eigvals(Matrix(pn.Az))
        V = eigvecs(Matrix(pn.Az))
        AbsAz = V*abs.(diagm(S))*inv(V)
        idx = findall(abs.(AbsAz) .> 1e-10)
        Iz = first.(Tuple.(idx)); Jz = last.(Tuple.(idx)); valsz = AbsAz[idx];
        AbsAz = sparse(Iz,Jz,Float64.(valsz),pn.nTotalEntries,pn.nTotalEntries);

        # compute normalized Legendre Polynomials
        Nq=200;
        (mu,w) = gauss(Nq);
        P=zeros(Nq,settings.nPN);
        for k=1:Nq
            PCurrent = collectPl(mu[k],lmax=settings.nPN-1);
            for i = 1:settings.nPN
                P[k,i] = PCurrent[i-1]/sqrt(gamma[i]);
            end
        end

        # setupt stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        gridSize = [nx, ny];
        gridWidth = [settings.dx,settings.dy];
        N = pn.nTotalEntries;
        L1x = spzeros(nx*ny,nx*ny);
        L1y = spzeros(nx*ny,nx*ny);
        L2x = spzeros(nx*ny,nx*ny);
        L2y = spzeros(nx*ny,nx*ny);

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3*(nx-2)*(ny-2)); J = zeros(3*(nx-2)*(ny-2)); vals = zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # x part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dx; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dx; 
                end
            end
        end
        L1x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(3*(nx-2)*(ny-2)); J .= zeros(3*(nx-2)*(ny-2)); vals .= zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # y part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dy; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dy; 
                end
            end
        end
        L1y = sparse(II,J,vals,nx*ny,nx*ny);

        II = zeros(2*(nx-2)*(ny-2)); J = zeros(2*(nx-2)*(ny-2)); vals = zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # x part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dx;
                end
            end
        end
        L2x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(2*(nx-2)*(ny-2)); J .= zeros(2*(nx-2)*(ny-2)); vals .= zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # y part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dy;
                end
            end
        end
        L2y = sparse(II,J,vals,nx*ny,nx*ny);

        sample = [1.0,0.0]; # Placeholder for the sample parameter

        uncertParam = "1";

        new(x,y,gridSize,gridWidth,settings,outRhs,gamma,AbsAx,AbsAz,P,mu,w,pn,L1x,L1y,L2x,L2y,uncertParam,sample);
    end
end

function setupIC(obj::solver)
    u = zeros(obj.settings.NCellsX,obj.settings.NCellsY,obj.pn.nTotalEntries);
    u[:,:,1] = IC(obj.settings,obj.sample,obj.uncertParam);
    u1 = zeros(obj.settings.NCellsX*obj.settings.NCellsY,obj.pn.nTotalEntries);
    for k = 1:obj.pn.nTotalEntries
        u1[:,k] = vec(u[:,:,k]);
    end
    return u1;
end

function K_step(obj,K,W,dt)
    WAzW = W'*obj.pn.Az'*W
    WAbsAzW = W'*obj.AbsAz'*W
    WAbsAxW = W'*obj.AbsAx'*W
    WAxW = W'*obj.pn.Ax'*W
    e1 = sparse([1],[1],[1.0],obj.pn.nTotalEntries,obj.pn.nTotalEntries); 
    E1 = zeros(obj.pn.nTotalEntries); E1[1] = 1.0;
    WeW = W'*e1*W;

    K = K .- dt*(obj.L2x*K*WAxW .+ obj.L2y*K*WAzW .+ obj.L1x*K*WAbsAxW .+ obj.L1y*K*WAbsAzW .+ obj.settings.sigmaT*K .- obj.settings.sigmaS*K*WeW .- obj.settings.Q*(E1'*W));
    return K;
end

function L_step(obj,X,Lt,dt)
    XL2xX = X'*obj.L2x*X;
    XL2yX = X'*obj.L2y*X;
    XL1xX = X'*obj.L1x*X;
    XL1yX = X'*obj.L1y*X;
    e1 = sparse([1],[1],[1.0],obj.pn.nTotalEntries,obj.pn.nTotalEntries); 
    E1 = zeros(obj.pn.nTotalEntries); E1[1] = 1.0;

    # L .= L .- dt*(obj.pn.Ax*L*XL2xX' .+ obj.pn.Az*L*XL2yX' .+ obj.AbsAx*L*XL1xX' .+ obj.AbsAz*L*XL1yX' .+ s.sigmaT*L .- s.sigmaS*e1*L);
    Lt .= Lt .- dt*(XL2xX*Lt*obj.pn.Ax' .+ XL2yX*Lt*obj.pn.Az' .+ XL1xX*Lt*obj.AbsAx' .+ XL1yX*Lt*obj.AbsAz' .+ (X'*obj.settings.sigmaT*X)*Lt .- (X'*obj.settings.sigmaS*X)*Lt*e1' .- X'*obj.settings.Q*E1');    
    return Lt;
end

function S_step(obj,X,S,W,Xh,Sh,Wh,dt)
    XL2xX = Xh'*obj.L2x*X
    XL2yX = Xh'*obj.L2y*X
    XL1xX = Xh'*obj.L1x*X
    XL1yX = Xh'*obj.L1y*X

    WAzW = W'*obj.pn.Az'*Wh
    WAbsAzW = W'*obj.AbsAz'*Wh
    WAbsAxW = W'*obj.AbsAx'*Wh
    WAxW = W'*obj.pn.Ax'*Wh

    e1 = sparse([1],[1],[1.0],obj.pn.nTotalEntries,obj.pn.nTotalEntries); 
    E1 = zeros(obj.pn.nTotalEntries); E1[1] = 1.0;
    
    WeW = W'*e1*Wh;

    S1 = S .- dt.*(XL2xX*Sh*WAxW .+ XL2yX*Sh*WAzW .+ XL1xX*Sh*WAbsAxW .+ XL1yX*Sh*WAbsAzW .+ (Xh'*obj.settings.sigmaT*X)*Sh*W'*Wh .- (Xh'*obj.settings.sigmaS*X)*Sh*WeW .- (Xh'*obj.settings.Q)*(E1'*Wh));
    return S1;
end

function pre_step()
    return nothing
end

function post_step()
    return nothing
end


function solveAugBUG(obj::solver)
    # Get rank
    r=50;
    rMaxTotal = Int(floor(obj.settings.r/2));
    s = obj.settings;
    # Set up initial condition and store as matrix
    u = setupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    # u = zeros(nx*ny,N);
    # for k = 1:N
    #     u[:,k] = vec(v[:,:,k]);
    # end

    nT = Int(ceil(s.Tend/s.dt))
    dt = s.dt

    prog = Progress(nT,1)

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    WAxW = zeros(r,r)
    WAzW = zeros(r,r)
    WAbsAxW = zeros(r,r)
    WAbsAzW = zeros(r,r)
    WeW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL1xX = zeros(r,r)
    XL1yX = zeros(r,r)

    MUp = zeros(2*r,r)
    NUp = zeros(2*r,r)

    XNew = zeros(nx*ny,r)
    STmp = zeros(r,r)

    e1 = sparse([1],[1],[1.0],N,N); 

    rankInTime = zeros(2,nT);
    NormInTime = zeros(2,nT);

    t = 0.0;

    for n=1:nT
        rankInTime[1,n] = t;
        rankInTime[2,n] = r;
        NormInTime[1,n] = t;
        NormInTime[2,n] = norm(S,2);

        ################## K-step ##################
        K = X*S;

        WAzW = W'*obj.pn.Az'*W
        WAbsAzW = W'*obj.AbsAz'*W
        WAbsAxW = W'*obj.AbsAx'*W
        WAxW = W'*obj.pn.Ax'*W
        WeW = W'*e1*W;

        K = K .- dt*(obj.L2x*K*WAxW .+ obj.L2y*K*WAzW .+ obj.L1x*K*WAbsAxW .+ obj.L1y*K*WAbsAzW .+ s.sigmaT*K .- s.sigmaS*K*WeW);

        K = [K X];
        XNew,STmp = qr!(K);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:2*r];

        MUp = XNew' * X;
        ################## L-step ##################
        L = W*S';

        XL2xX = X'*obj.L2x*X
        XL2yX = X'*obj.L2y*X
        XL1xX = X'*obj.L1x*X
        XL1yX = X'*obj.L1y*X

        L .= L .- dt*(obj.pn.Ax*L*XL2xX' .+ obj.pn.Az*L*XL2yX' .+ obj.AbsAx*L*XL1xX' .+ obj.AbsAz*L*XL1yX' .+ L*X's.sigmaT*X .- e1*L*X's.sigmaS*X);
                
        L = [L W];
        WNew,STmp = qr(L);
        WNew = Matrix(WNew)
        WNew = WNew[:,1:2*r];

        NUp = WNew' * W;
        W = WNew;
        X = XNew;
        ################## S-step ##################
        S = MUp*S*(NUp')

        XL2xX = X'*obj.L2x*X
        XL2yX = X'*obj.L2y*X
        XL1xX = X'*obj.L1x*X
        XL1yX = X'*obj.L1y*X

        WAzW = W'*obj.pn.Az'*W
        WAbsAzW = W'*obj.AbsAz'*W
        WAbsAxW = W'*obj.AbsAx'*W
        WAxW = W'*obj.pn.Ax'*W
        WeW = W'*e1*W;

        S .= S .- dt.*(XL2xX*S*WAxW .+ XL2yX*S*WAzW .+ XL1xX*S*WAbsAxW .+ XL1yX*S*WAbsAzW .+ X'*s.sigmaT*X*S .- X'*s.sigmaS*X*S*WeW);
        
        ################## truncate ##################

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(S);
        U = Matrix(U); V = Matrix(V)
        rmax = -1;
        S .= zeros(size(S));

        tmp = 0.0;
        tol = obj.settings.epsAdapt*norm(D);
        
        rmax = Int(floor(size(D,1)/2));
        
        for j=1:2*rmax
            tmp = sqrt(sum(D[j:2*rmax]).^2);
            if(tmp<tol)
                rmax = j;
                break;
            end
        end
        
        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,2);

        for l = 1:rmax
            S[l,l] = D[l];
        end

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        # update solution with new rank
        XNew = XNew*U;
        WNew = WNew*V;

        # update solution with new rank
        S = S[1:rmax,1:rmax];
        X = XNew[:,1:rmax];
        W = WNew[:,1:rmax];

        # update rank
        r = rmax;
        
        t += dt;
    end

    # return end time and solution
    return sqrt(2).*X*S*W',rankInTime;

end

function solveAugBUG(obj::solver,sample::Array{Float64,1})
    obj.sample = sample;
    g,rVec = solveAugBUG(obj);
    return g,rVec;
end


function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*nx + j]
        end
    end
    return m;
end