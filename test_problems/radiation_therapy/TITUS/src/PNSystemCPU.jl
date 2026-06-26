__precompile__

mutable struct PNSystemCPU{T<:AbstractFloat}
    # symmetric flux matrices
    Ax::SparseMatrixCSC{T, Int32};
    Ay::SparseMatrixCSC{T, Int32};
    Az::SparseMatrixCSC{T, Int32};

    # Solver settings
    settings::Settings;

    # total number of moments
    nTotalEntries::Int
    # degree of expansion
    N::Int

    M::SparseMatrixCSC{ComplexF64, Int};

    T::DataType;

    # Roe matrices
    AbsAx::Array{T, 2};
    AbsAy::Array{T, 2};
    AbsAz::Array{T, 2};

    # decomposed flux matrices
    Vx::Array{T, 2}
    Λx⁺::Diagonal{T, Array{T, 1}}
    Λx⁻::Diagonal{T, Array{T, 1}}

    Vy::Array{T, 2}
    Λy⁺::Diagonal{T, Array{T, 1}}
    Λy⁻::Diagonal{T, Array{T, 1}}

    Vz::Array{T, 2}
    Λz⁺::Diagonal{T, Array{T, 1}}
    Λz⁻::Diagonal{T, Array{T, 1}}

    # constructor
    function PNSystemCPU(settings::Settings,T::DataType=Float64)
        N = settings.nPN;
        nTotalEntries = GlobalIndex( N, N ) + 1;    # total number of entries for sytem matrix

        IndI = []; IndJ = []; val = [];
        # Assemble transformation matrix
        for m = 2:N+1
            for l = 1:m-1
                r = (m.-1)^2 .+2*l;
                IndItmp = Int.([r.-1,r,r-1,r]);
                IndJtmp = Int.([(m-1)^2+l,(m-1)^2+l,m^2+1-l,m^2+1-l]);
                valTmp = [1,-1im,(-1).^(m+l),(-1).^(m+l)*1im]./sqrt(2);
                IndI = [IndI;IndItmp];
                IndJ = [IndJ;IndJtmp];
                val = [val;valTmp]
            end
        end
        for m = 1:1:N+1
            IndI = [IndI;m^2];
            IndJ = [IndJ;(m-1)^2+m];
            val = [val;1]
        end
        M = sparse(Int.(IndI),Int.(IndJ),val,nTotalEntries,nTotalEntries);

        # allocate dummy matrices. Correct matrices are reallocated in respective functions
        Ax = sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries));
        Ay = sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries));
        Az = sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries));
        new{T}(Ax,Ay,Az,settings,nTotalEntries,N,M,T);
    end
end

function SetupSystemMatrices(obj::PNSystemCPU)
    nTotalEntries = obj.nTotalEntries;    # total number of entries for sytem matrix
    N = obj.N

    Ax = zeros(nTotalEntries,nTotalEntries)
    Ay = zeros(nTotalEntries,nTotalEntries)
    Az = zeros(nTotalEntries,nTotalEntries)

    # loop over columns of A
    for l = 0:N
        for k=-l:l
            i = GlobalIndex( l, k ) ;

            # flux matrix in direction x
            if k != -1
                j = GlobalIndex( l - 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ax[i+1,j+1] = 0.5 * CTilde( l - 1, abs( k ) - 1 ); end

                j = GlobalIndex( l + 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ax[i+1,j+1] = -0.5 * DTilde( l + 1, abs( k ) - 1 ); end
            end

            j = GlobalIndex( l - 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ax[i+1,j+1] = -0.5 * ETilde( l - 1, abs( k ) + 1 ); end

            j = GlobalIndex( l + 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ax[i+1,j+1] = 0.5 * FTilde( l + 1, abs( k ) + 1 ); end

            # flux matrix in direction y
            if  k != 1
                j = GlobalIndex( l - 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = -0.5 * Sgn( k ) * CTilde( l - 1, abs( k ) - 1 ); end

                j = GlobalIndex( l + 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = 0.5 * Sgn( k ) * DTilde( l + 1, abs( k ) - 1 ); end
            end

            j = GlobalIndex( l - 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = -0.5 * Sgn( k ) * ETilde( l - 1, abs( k ) + 1 ); end

            j = GlobalIndex( l + 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = 0.5 * Sgn( k ) * FTilde( l + 1, abs( k ) + 1 ); end

            # flux matrix in direction z
            j = GlobalIndex( l - 1, k );
            if  j >= 0 && j < nTotalEntries  Az[i+1,j+1] = AParam( l - 1, k ); end

            j = GlobalIndex( l + 1, k );
            if  j >= 0 && j < nTotalEntries  Az[i+1,j+1] = BParam( l + 1, k ); end
        end
    end
    return Ax,-Ay,-Az # swap sign to match our definition of the real spherical harmonics in utilities.jl
end

function SetupSystemMatricesSparse(obj::PNSystemCPU)
    nTotalEntries = obj.nTotalEntries;    # total number of entries for sytem matrix
    N = obj.N

    Ix = []; Jx = []; valsx = [];
    Iy = []; Jy = []; valsy = [];
    Iz = []; Jz = []; valsz = [];

    # loop over columns of A
    for l = 0:N
        for k=-l:l
            i = GlobalIndex( l, k ) ;

            # flux matrix in direction x
            if k != -1
                j = GlobalIndex( l - 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ix = [Ix;i+1]; Jx = [Jx;j+1]; valsx = [valsx; 0.5 * CTilde( l - 1, abs( k ) - 1 )]; end

                j = GlobalIndex( l + 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ix = [Ix;i+1]; Jx = [Jx;j+1]; valsx = [valsx; -0.5 * DTilde( l + 1, abs( k ) - 1 )]; end
            end

            j = GlobalIndex( l - 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ix = [Ix;i+1]; Jx = [Jx;j+1]; valsx = [valsx; -0.5 * ETilde( l - 1, abs( k ) + 1 )]; end

            j = GlobalIndex( l + 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ix = [Ix;i+1]; Jx = [Jx;j+1]; valsx = [valsx; 0.5 * FTilde( l + 1, abs( k ) + 1 )]; end

            # flux matrix in direction y
            if  k != 1
                j = GlobalIndex( l - 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries Iy = [Iy;i+1]; Jy = [Jy;j+1]; valsy = [valsy; -0.5 * Sgn( k ) * CTilde( l - 1, abs( k ) - 1 )]; end

                j = GlobalIndex( l + 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries  Iy = [Iy;i+1]; Jy = [Jy;j+1]; valsy = [valsy; 0.5 * Sgn( k ) * DTilde( l + 1, abs( k ) - 1 )]; end
            end

            j = GlobalIndex( l - 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Iy = [Iy;i+1]; Jy = [Jy;j+1]; valsy = [valsy; -0.5 * Sgn( k ) * ETilde( l - 1, abs( k ) + 1 )]; end

            j = GlobalIndex( l + 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Iy = [Iy;i+1]; Jy = [Jy;j+1]; valsy = [valsy; 0.5 * Sgn( k ) * FTilde( l + 1, abs( k ) + 1 )]; end

            # flux matrix in direction z
            j = GlobalIndex( l - 1, k );
            if  j >= 0 && j < nTotalEntries  Iz = [Iz;i+1]; Jz = [Jz;j+1]; valsz = [valsz; AParam( l - 1, k )]; end

            j = GlobalIndex( l + 1, k );
            if  j >= 0 && j < nTotalEntries  Iz = [Iz;i+1]; Jz = [Jz;j+1]; valsz = [valsz; BParam( l + 1, k )]; end
        end
    end
    obj.Ax = sparse(Ix,Jx,- obj.T.(valsx),obj.nTotalEntries,obj.nTotalEntries); # swap sign to match our definition of the real spherical harmonics in utilities.jl
    obj.Ay = sparse(Iy,Jy,- obj.T.(valsy),obj.nTotalEntries,obj.nTotalEntries); # swap sign to match our definition of the real spherical harmonics in utilities.jl
    obj.Az = sparse(Iz,Jz,obj.T.(valsz),obj.nTotalEntries,obj.nTotalEntries);
end

function SetupRoeMatrices(obj::PNSystemCPU)
    T = obj.T
    eig = LinearAlgebra.eigen(obj.Ax)
    S = eig.values
    V = eig.vectors
    obj.AbsAx = V*abs.(Diagonal(S))*inv(V)

    eig = LinearAlgebra.eigen(obj.Ay)
    S = eig.values
    V = eig.vectors
    obj.AbsAy = V*abs.(diagm(S))*inv(V)
    
    eig = LinearAlgebra.eigen(obj.Az)
    S = eig.values
    V = eig.vectors
    obj.AbsAz = V*abs.(diagm(S))*inv(V)
end
