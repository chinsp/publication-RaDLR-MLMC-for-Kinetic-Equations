__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum
using Base.Threads
using Interpolations
using TimerOutputs
using Random, Distributions
using Base.Threads: SpinLock
using CUDA
include("raytracer/proton_data.jl")
using .ProtonData
include("raytracer/uncollided.jl")
using .Uncollided
include("stencils.jl")
include("raytracer/raytracer.jl")
using .UniformRayTracer

mutable struct SolverGPU{T<:AbstractFloat}
    # spatial grid of cell interfaces
    x::Array{T};
    y::Array{T};
    z::Array{T};

    order::Int;
    
    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{T,1};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # stencil matrices
    stencil::UpwindStencil3DCUDA;

    # material density
    densityVec::Array{T,1};
    densityVecHU::Array{Int,1};

    # dose vector
    dose::Array{T,1};

    boundaryIdx::Array{Int,1}

    Q::Quadrature
    O::Array{T,2};
    M::Array{T,2};

    T::DataType;

    OReduced::Array{T,2};
    MReduced::Array{T,2};
    qReduced::Array{T,2};

    # constructor
    function SolverGPU(settings,order=2)
        T = Float32; # define accuracy 
        x = settings.x;
        y = settings.y;
        z = settings.z;

        nx = settings.NCellsX;
        ny = settings.NCellsY;
        nz = settings.NCellsZ;

        # setup flux matrix
        gamma = zeros(T,settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        @timeit to "CSD and MaterialParameters" begin
            # construct CSD fields
            csd = CSD(settings,T);
        end

        # allocate dose vector
        dose = zeros(T,nx*ny*nz)
        @timeit to "Set up Pn system" begin
            @timeit to "Constructor" begin
                pn = PNSystem(settings,T)
            end
            @timeit to "Sparse system matrices" begin
                SetupSystemMatricesSparse(pn)
            end
        end
        @timeit to "Set up stencil" begin
            stencil = UpwindStencil3DCUDA(settings,order);
        end
        Norder = (settings.nPN+1)^2
        
        @timeit to "Boundary indices" begin
            # collect boundary indices
            if order == 1
                boundaryIdx = zeros(Int,2*nx*ny+2*ny*nz + 2*nx*nz)
                Threads.@threads for i = 1:nx
                    for k = 1:nz
                        j = 1;
                        boundaryIdx[(i-1)*nz*2+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        j = ny;
                        boundaryIdx[(i-1)*nz*2+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for i = 1:nx
                    for j = 1:ny
                        k = 1;
                        boundaryIdx[2*nx*nz+2*(i-1)*ny+(j-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        k = nz;
                        boundaryIdx[2*nx*nz+2*(i-1)*ny+(j-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
        
                Threads.@threads for j = 1:ny
                    for k = 1:nz
                        i = 1;
                        boundaryIdx[2*nx*ny+2*nx*nz+2*(j-1)*nz+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        i = nx;
                        boundaryIdx[2*nx*ny+2*nx*nz+2*(j-1)*nz+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
            elseif order == 2
                boundaryIdx = zeros(Int,4*nx*ny+4*ny*nz+4*nx*nz)
                counter = 0;
                Threads.@threads for i = 1:nx
                    Threads.@threads for k = 1:nz
                        j = 1;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        j = 2;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+2] = vectorIndex(nx,ny,i,j,k)
                        j = ny;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        j = ny-1;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for i = 1:nx
                    Threads.@threads for j = 1:ny
                        k = 1;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        k = 2;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+2] = vectorIndex(nx,ny,i,j,k)
                        k = nz;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        k = nz - 1;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for j = 1:ny
                    Threads.@threads for k = 1:nz
                        i = 1;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        i = 2;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+2] = vectorIndex(nx,ny,i,j,k);
                        i = nx;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        i = nx - 1;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
            end
        end
        @timeit to "Quadrature and Trafo matrices" begin
            # setup quadrature
            qorder = 1 
            if iseven(qorder) qorder += 1; end 
            qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
            Q = Quadrature(qorder,qtype);

            O,M = ComputeTrafoMatrices(Q,Norder,settings.nPN,[settings.Omega1 settings.Omega2 settings.Omega3]);
        end

        densityVec = T.(settings.density[:]);
        densityVecHU = settings.densityHU[:];

        new{T}(T.(x),T.(y),T.(z),order,settings,gamma,csd,pn,stencil,densityVec,densityVecHU,dose,boundaryIdx,Q,T.(O),T.(M),T);
    end
end

function implicit_L_step!(L::CuMatrix{T},
                            d::CuVector{T},
                            B::CuMatrix{T},
                            dt::T) where {T<:AbstractFloat}

    m, r = size(L)
    @assert length(d) == m
    Bt = transpose(B)
    d_cpu = Array(d)
    unique_d = Base.unique(d_cpu)
    nd = length(unique_d)

    d_to_factor = Dict{T, Tuple{CuMatrix{T}, CuMatrix{T}}}()

    for dj in unique_d
        I_r_gpu = CUDA.CuArray(Matrix{T}(I, r, r))
        A = I_r_gpu .- dt * dj * Bt
        F = lu(A)

        U = UpperTriangular(F.U)
        Lfac = UnitLowerTriangular(F.L)

        d_to_factor[dj] = (U, Lfac)
    end

    groups = Dict{T, Vector{Int}}()
    for i in 1:m
        push!(get!(groups, d_cpu[i], Int[]), i)
    end

    # Solve (I - dt * d_i * Bt)ᵀ * L[i, :] = L_old[i, :] for each group 
    for (dj, idx) in groups
        U, Lfac = d_to_factor[dj]
        rows = @view L[idx, :]
        rows_t = permutedims(rows)    

        # forward solve  
        CUDA.CUBLAS.trsm!('L','L','N','U', one(T), Lfac, rows_t)
        # backward solve
        CUDA.CUBLAS.trsm!('L','U','N','N', one(T), U, rows_t)

        transpose!(rows, rows_t)
    end

    return L
end

function Solve_sample(obj::SolverGPU{T},alpha::Vector{T}=[0.0 0.0]) where {T<:AbstractFloat}
    nB = length(obj.settings.Omega1)
    alpha_xyz = zeros(nB,3)

    #rotate shifts to xyz and apply
    for b=1:nB
        alpha_xyz[b,:] = rotate_bev_to_xyz([obj.settings.Omega1[b],obj.settings.Omega2[b],obj.settings.Omega3[b]])*[alpha[1],alpha[2],0]
        obj.settings.x0[b] += alpha_xyz[b,1]
        obj.settings.y0[b] += alpha_xyz[b,2]
        obj.settings.z0[b] += alpha_xyz[b,3]
    end
    _,_,_,_, dose, _, rVec, _ = Solve(obj, "Boltzmann",true, "Boltzmann")
    
    #go back to old starting position in case settings are reused for another computation
    for b=1:nB
        obj.settings.x0[b] -= alpha_xyz[b,1]
        obj.settings.y0[b] -= alpha_xyz[b,2]
        obj.settings.z0[b] -= alpha_xyz[b,3]
    end

    #return normaised dose
    dV = obj.settings.dx * obj.settings.dy * obj.settings.dz
    dose[dose.<0] .= 0.0
    return dose ./ (sum(dose) * dV) 

end

function Solve(obj::SolverGPU{T}, model::String="Boltzmann",
                                           trace::Bool=false,
                                           model_collided::String="Boltzmann") where {T<:AbstractFloat}

    r     = Int(floor(obj.settings.r / 2))
    order = obj.order

    energy = obj.csd.eGrid
    nx, ny, nz = obj.settings.NCellsX, obj.settings.NCellsY, obj.settings.NCellsZ
    N = obj.pn.nTotalEntries
    s = obj.settings

    @timeit to "Ray-tracer" begin
        E_tracer, psiE = RunTracer_UniDirectional(obj, model, trace)
    end

    q        = [obj.settings.Omega1 obj.settings.Omega2 obj.settings.Omega3]
    MReduced = CuArray(T.(obj.M))
    M1       = MReduced[1,:]
    nq       = size(q, 1)
    e1       = CuArray(T.(I(N)[1,:]))

    X, _, _ = svd(zeros(T, nx*ny*nz, r))
    W, _, _ = svd(zeros(T, N, r))
    X = CuArray(X[:,1:r])
    W = CuArray(W[:,1:r])
    S = CUDA.zeros(T, r, r)

    X[obj.boundaryIdx,:] .= 0.0

    nEnergies       = length(energy)
    dE              = energy[1] - energy[2]
    obj.settings.dE = dE

    # Safe parallel construction of the material-index lookup table.
    idx  = Base.unique(i -> obj.densityVec[i], 1:length(obj.densityVec))
    idxK = Vector{Vector{Int64}}(undef, length(idx))
    Threads.@threads for k in eachindex(idx)
        idxK[k] = findall(==(obj.densityVec[idx[k]]), obj.densityVec)
    end

    println("CFL = ", dE / min(s.dx, s.dy, s.dz))

    prog = Progress(nEnergies-1, 1)
    rVec = r .* ones(2, nEnergies)

    dose_coll = CUDA.zeros(T, nx*ny*nz)
    dose      = CuArray(obj.dose)

    @timeit to "Setup upwind stencil" begin
        stencil = UpwindStencil3DCUDA(s, order)
        D⁺₁, D⁻₁ = stencil.D⁺₁, stencil.D⁻₁
        D⁺₂, D⁻₂ = stencil.D⁺₂, stencil.D⁻₂
        D⁺₃, D⁻₃ = stencil.D⁺₃, stencil.D⁻₃
    end

    CUDA.reclaim(); 

    @timeit to "Setup Pn system" begin
        @timeit to "Eigendecomposition Ax" begin
            Σ₁, T₁ = eigen(CuArray(obj.pn.Ax))
            T₁⁻¹ = T₁'
            Σ₁⁺, Σ₁⁻ = max.(Σ₁, 0), min.(Σ₁, 0); Σ₁ = nothing
        end
        @timeit to "Eigendecomposition Ay" begin
            Σ₂, T₂ = eigen(CuArray(obj.pn.Ay))
            T₂⁻¹ = T₂'
            Σ₂⁺, Σ₂⁻ = max.(Σ₂, 0), min.(Σ₂, 0); Σ₂ = nothing
        end
        @timeit to "Eigendecomposition Az" begin
            Σ₃, T₃ = eigen(CuArray(obj.pn.Az))
            T₃⁻¹ = T₃'
            Σ₃⁺, Σ₃⁻ = max.(Σ₃, 0), min.(Σ₃, 0); Σ₃ = nothing
        end
    end

    ∫Y₀⁰dΩ = T(4π / sqrt(4π))

    @timeit to "Interpolate to energy grid" begin
        nPsi = size(psiE, 1)
        nB   = size(psiE, 3)
        psi  = CUDA.zeros(T, nx*ny*nz, nB)

        # The FP tracer shifts the first energy point slightly to avoid an
        # interpolation boundary issue at the upper end of the interval.
        if model == "FP"
            E_tracer[1] = E_tracer[1] .- 0.001
        end

        ETracer2E = if nB == 1
            interpolate((1:nPsi, E_tracer[1:end]), psiE[:,:,1],
                        (NoInterp(), Gridded(Linear())))
        else
            interpolate((1:nPsi, E_tracer[1:end], 1:nB), psiE,
                        (NoInterp(), Gridded(Linear()), NoInterp()))
        end
        psiE = nothing
    end

    # CPU staging buffers for stopping-power arrays; reused each iteration
    # to avoid per-step heap allocations.
    Sinv_CPU    = zeros(T, nPsi)
    SinvMid_CPU = zeros(T, nPsi)
    SinvEnd_CPU = zeros(T, nPsi)

    # GPU stopping-power vectors; updated in-place each iteration via copyto!
    Sinv    = CUDA.zeros(T, nPsi)
    SinvMid = CUDA.zeros(T, nPsi)
    SinvEnd = CUDA.zeros(T, nPsi)

    wMat = T.(CuArray(matComp(s.densityHU[:]) .* s.density[:]' ./ 100))
    Nmat = size(wMat, 1)

    # Preallocated rank-doubling buffers for the SVD augmentation step.
    XK_buf = CUDA.zeros(T, nx*ny*nz, 2r)
    WL_buf = CUDA.zeros(T, N, 2r)

    # Maps an RK4 substep index to the corresponding stopping-power vector.
    # Index 1 = interval start, 2 = midpoint, 3 = interval end, matching the
    # convention used inside rk4_idx when it calls f(t, u, idx).
    sinv_at(idx) = idx == 1 ? Sinv : idx == 2 ? SinvMid : SinvEnd

    # Computes an orthonormal basis for the column space of [A  B] using a
    # preallocated buffer, and returns the basis together with the projection
    # MUp = basis' * A needed for the subsequent S-step update.
    function augmented_svd!(buf, A, B)
        rc = size(A, 2)
        buf[:, 1:rc]     .= A
        buf[:, rc+1:2rc] .= B
        Atmp, _, _ = svd(view(buf, :, 1:2rc))
        return Atmp, Atmp' * A
    end

    #=function augmented_svd!(buf, A, B)
        rc  = size(A, 2)
        buf[:, 1:rc]     .= A
        buf[:, rc+1:2rc] .= B
        F    = qr(view(buf, :, 1:2rc))
        Atmp = CuArray(F.Q)
        return Atmp, Atmp' * A
    end=#

    # Recomputes the six W-dependent Gram matrices that appear in the K- and
    # S-step right-hand sides. 
    function compute_W_grams()
        WT1Sp = (W'*T₁) * (Σ₁⁺ .* (T₁⁻¹*W))
        WT1Sm = (W'*T₁) * (Σ₁⁻ .* (T₁⁻¹*W))
        WT2Sp = (W'*T₂) * (Σ₂⁺ .* (T₂⁻¹*W))
        WT2Sm = (W'*T₂) * (Σ₂⁻ .* (T₂⁻¹*W))
        WT3Sp = (W'*T₃) * (Σ₃⁺ .* (T₃⁻¹*W))
        WT3Sm = (W'*T₃) * (Σ₃⁻ .* (T₃⁻¹*W))
        return WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm
    end

    # K-step right-hand sides.
    function FKx(K, idx, WT1Sp, WT1Sm)
        Sv = sinv_at(idx)
        return -(D⁺₁*(Sv.*K)*WT1Sp .+ D⁻₁*(Sv.*K)*WT1Sm)
    end
    function FKy(K, idx, WT2Sp, WT2Sm)
        Sv = sinv_at(idx)
        return -(D⁺₂*(Sv.*K)*WT2Sp .+ D⁻₂*(Sv.*K)*WT2Sm)
    end
    function FKz(K, idx, WT3Sp, WT3Sm)
        Sv = sinv_at(idx)
        return -(D⁺₃*(Sv.*K)*WT3Sp .+ D⁻₃*(Sv.*K)*WT3Sm)
    end
    function FK(K, idx, WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm)
        Sv = sinv_at(idx)
        SvK = Sv.*K
        return -(D⁺₁*SvK*WT1Sp .+ D⁻₁*SvK*WT1Sm)-(D⁺₂*SvK*WT2Sp .+ D⁻₂*SvK*WT2Sm)-(D⁺₃*SvK*WT3Sp .+ D⁻₃*SvK*WT3Sm)
    end

    # L-step right-hand sides.
    function FLx(L, idx)
        Sv = sinv_at(idx)
        return -(Σ₁⁺.*L*(X'*D⁺₁*(Sv.*X))' .+ Σ₁⁻.*L*(X'*D⁻₁*(Sv.*X))')
    end
    function FLy(L, idx)
        Sv = sinv_at(idx)
        return -(Σ₂⁺.*L*(X'*D⁺₂*(Sv.*X))' .+ Σ₂⁻.*L*(X'*D⁻₂*(Sv.*X))')
    end
    function FLz(L, idx)
        Sv = sinv_at(idx)
        return -(Σ₃⁺.*L*(X'*D⁺₃*(Sv.*X))' .+ Σ₃⁻.*L*(X'*D⁻₃*(Sv.*X))')
    end

    # S-step right-hand sides.
    function FSx(S_mat, idx, WT1Sp, WT1Sm)
        Sv     = sinv_at(idx)
        XtD1pX = X' * D⁺₁ * (Sv .* X)
        XtD1mX = X' * D⁻₁ * (Sv .* X)
        return -(XtD1pX * S_mat * WT1Sp .+ XtD1mX * S_mat * WT1Sm)
    end
    function FSy(S_mat, idx, WT2Sp, WT2Sm)
        Sv     = sinv_at(idx)
        XtD2pX = X' * D⁺₂ * (Sv .* X)
        XtD2mX = X' * D⁻₂ * (Sv .* X)
        return -(XtD2pX * S_mat * WT2Sp .+ XtD2mX * S_mat * WT2Sm)
    end
    function FSz(S_mat, idx, WT3Sp, WT3Sm)
        Sv     = sinv_at(idx)
        XtD3pX = X' * D⁺₃ * (Sv .* X)
        XtD3mX = X' * D⁻₃ * (Sv .* X)
        return -(XtD3pX * S_mat * WT3Sp .+ XtD3mX * S_mat * WT3Sm)
    end
    function FS(S_mat, idx, WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm)
        Sv     = sinv_at(idx)
        SvX = Sv .* X
        XtD1pX = X' * D⁺₁ * SvX
        XtD1mX = X' * D⁻₁ * SvX
        XtD2pX = X' * D⁺₂ * SvX
        XtD2mX = X' * D⁻₂ * SvX
        XtD3pX = X' * D⁺₃ * SvX
        XtD3mX = X' * D⁻₃ * SvX
        return -(XtD1pX * S_mat * WT1Sp .+ XtD1mX * S_mat * WT1Sm)-(XtD2pX * S_mat * WT2Sp .+ XtD2mX * S_mat * WT2Sm)-(XtD3pX * S_mat * WT3Sp .+ XtD3mX * S_mat * WT3Sm)
    end

    stoppingPowerOld = get_S_at_energy(obj.csd, energy[1], obj.settings.density[:])

    println("Starting energy loop")

    @timeit to "DLRA" begin
    for n = 2:nEnergies

        @timeit to "Stopping power + interp" begin
            dE     = energy[n-1] - energy[n]
            dEGrid = dE

            stoppingPowerCurrent = get_S_at_energy(obj.csd, energy[n], obj.settings.density[:])
            stoppingPowerMid = get_S_at_energy(obj.csd, energy[n-1] - T(0.5 * dE), obj.settings.density[:])
            for j = 1:length(idx)
                Sinv_CPU[idxK[j]]    .= 1 ./ stoppingPowerOld[j]
                SinvMid_CPU[idxK[j]] .= 1 ./ stoppingPowerMid[j]
                SinvEnd_CPU[idxK[j]] .= 1 ./ stoppingPowerCurrent[j]
            end
            stoppingPowerOld .= stoppingPowerCurrent
            copyto!(Sinv,    Sinv_CPU)
            copyto!(SinvMid, SinvMid_CPU)
            copyto!(SinvEnd, SinvEnd_CPU)
        end

        @timeit to "Scattering coeffs (Dvec)" begin
            if model_collided == "FP"
                xi     = T.(XiAtEnergyandX(obj.csd, energy[n]))
                sigmaS = zeros(1, Nmat)
            else
                sigmaS = SigmaAtEnergyandX(obj.csd, energy[n])
            end

            DvecCPU = zeros(T, obj.pn.nTotalEntries, Nmat)
            if model_collided == "FP"
                N_corr = 19
                for j = 1:Nmat
                    for l = 0:obj.pn.N, k = -l:l
                        DvecCPU[GlobalIndex(l,k)+1, j] = 0.5 * xi[1,j] * (N_corr*(N_corr+1) - l*(l+1))
                    end
                    sigmaS[1,j] = 0.5 * xi[1,j] * (N_corr*(N_corr+1))
                end
            else
                for j = 1:Nmat, l = 0:obj.pn.N, k = -l:l
                    DvecCPU[GlobalIndex(l,k)+1, j] = sigmaS[l+1, j]
                end
            end
            Dvec = CuArray(T.(DvecCPU))
        end

        @timeit to "Uncollided flux interp" begin
            if nB == 1
                psi .= CuArray(ETracer2E.(1:nPsi, energy[n]))
            else
                for b = 1:nB
                    psi[:,b] .= CuArray(ETracer2E.(1:nPsi, energy[n], b))
                end
            end
        end

        if n > 2
            r_cur = size(X, 2)
            if 2r_cur > size(XK_buf, 2)
                XK_buf = CUDA.zeros(T, nx*ny*nz, 2r_cur)
                WL_buf = CUDA.zeros(T, N, 2r_cur)
            end

            @timeit to "W gram matrices (pre-K)" begin
                WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm = compute_W_grams()
            end

            @timeit to "K-step" begin
                X[obj.boundaryIdx,:] .= 0.0
                K  = X * S
                K .= rk4_idx(dE, (t, K_, idx) -> FK(K_, idx, WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm), K)
                #K .= rk4_idx(dE, (t, K_, idx) -> FKx(K_, idx, WT1Sp, WT1Sm), K)
                #K .= rk4_idx(dE, (t, K_, idx) -> FKy(K_, idx, WT2Sp, WT2Sm), K)
                #K .= rk4_idx(dE, (t, K_, idx) -> FKz(K_, idx, WT3Sp, WT3Sm), K)
                @timeit to "K-step QR" begin
                    Xtmp, MUp = augmented_svd!(XK_buf, X, K)
                end
            end

            @timeit to "L-step" begin
                L  = T₁⁻¹ * W * S'
                L .= T₁ * rk4_idx(dE, (t, L_, idx) -> FLx(L_, idx), L)
                L .= T₂⁻¹ * L
                L .= T₂ * rk4_idx(dE, (t, L_, idx) -> FLy(L_, idx), L)
                L .= T₃⁻¹ * L
                L .= T₃ * rk4_idx(dE, (t, L_, idx) -> FLz(L_, idx), L)
                Wtmp, NUp = augmented_svd!(WL_buf, W, L)
            end

            X, W = Xtmp, Wtmp
            X[obj.boundaryIdx,:] .= 0.0

            @timeit to "W gram matrices (pre-S)" begin
                WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm = compute_W_grams()
            end

            @timeit to "S-step" begin
                S  = MUp * S * NUp'
                S .= rk4_idx(dE, (t, S_, idx) -> FS(S_, idx, WT1Sp, WT1Sm, WT2Sp, WT2Sm, WT3Sp, WT3Sm), S)
                #S .= rk4_idx(dE, (t, S_, idx) -> FSx(S_, idx, WT1Sp, WT1Sm), S)
                #S .= rk4_idx(dE, (t, S_, idx) -> FSy(S_, idx, WT2Sp, WT2Sm), S)
                #S .= rk4_idx(dE, (t, S_, idx) -> FSz(S_, idx, WT3Sp, WT3Sm), S)
            end

        end

        @timeit to "Out-scattering" begin
            # if model_collided == "FP"
                @timeit to "Truncation (post-transport)" begin
                    X, S, W = truncateCUDANew(obj, T.(X), T.(S), T.(W))
                    r = size(S, 1)
                    if 2r > size(XK_buf, 2)
                        XK_buf = CUDA.zeros(T, nx*ny*nz, 2r)
                        WL_buf = CUDA.zeros(T, N, 2r)
                    end
                end
                L  = W * S'
                L0 = copy(L)
                for j = 1:Nmat
                    implicit_L_step!(L, T.(Dvec[:,j] .- sigmaS[1,j]),
                                     X' * (wMat[j,:] .* SinvEnd .* X), dE)
                end
            # else
            #     L  = W * S'
            #     L0 = copy(L)
            #     for j = 1:Nmat
            #         L .+= dE .* (Dvec[:,j] .- sigmaS[1,j]) .* (L0 * (X' * (wMat[j,:] .* SinvEnd .* X)))
            #     end
            # end
            W, S1, S2 = svd(L)
            S .= S2 * Diagonal(S1)
        end

        @timeit to "In-scattering" begin
            r_cur = size(X, 2)
            if 2r_cur > size(XK_buf, 2)
                XK_buf = CUDA.zeros(T, nx*ny*nz, 2r_cur)
                WL_buf = CUDA.zeros(T, N, 2r_cur)
            end

            X[obj.boundaryIdx,:] .= 0.0
            K = X * S
            for j = 1:Nmat
                K .+= dE .* wMat[j,:] .* SinvEnd .* psi * MReduced' * (Dvec[:,j] .* W)
            end
            K[obj.boundaryIdx,:] .= 0.0
            Xtmp, MUp = augmented_svd!(XK_buf, X, K)

            L = W * S'
            for j = 1:Nmat
                L .+= dE .* Dvec[:,j] .* MReduced * (X' * (wMat[j,:] .* SinvEnd .* psi))'
            end
            Wtmp, NUp = augmented_svd!(WL_buf, W, L)
 
            X, W = Xtmp, Wtmp
            S  = MUp * S * NUp'
            for j = 1:Nmat
                S .+= dE .* (X' * (wMat[j,:] .* SinvEnd .* psi)) * MReduced' * (Dvec[:,j] .* W)
            end
        end

        @timeit to "Dose accumulation" begin
            coll_flux  = X * S * (W' * e1)
            scale      = dEGrid * ∫Y₀⁰dΩ
            dose      .+= scale .* (coll_flux .+ psi * M1)
            dose_coll .+= scale .* coll_flux
        end

        @timeit to "Truncation (final)" begin
            X, S, W = truncateCUDANew(obj, T.(X), T.(S), T.(W))
            r = size(S, 1)
            if 2r > size(XK_buf, 2)
                XK_buf = CUDA.zeros(T, nx*ny*nz, 2r)
                WL_buf = CUDA.zeros(T, N, 2r)
            end
        end

        rVec[1, n] = energy[n]
        rVec[2, n] = r
        ProgressMeter.next!(prog)
    end
    end

    U, Sigma, V = svd(Matrix(S))
    return Matrix(X)*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*Matrix(W)*V,
           Matrix(W)*V, Vector(dose), Vector(dose_coll), rVec, Matrix(psi)
end

function RunTracer_UniDirectional(obj::SolverGPU{T}, model::String, trace::Bool; coords_bounds::Matrix{Float64}=zeros(0,0)) where {T<:AbstractFloat}
    ## this function has been severely reduced since we only load precomputed tracer results in this version
    nE = 1
    E_tracer = []
    tracerDirs = [obj.settings.Omega1 obj.settings.Omega2 obj.settings.Omega3]
    nB = size(tracerDirs,1)
    trace_mode = :midpoints #default
    @timeit to "Ray-tracer" begin

        if trace 
            Ncells = [obj.settings.NCellsX,obj.settings.NCellsY,obj.settings.NCellsZ]
            bounds_min = [obj.settings.a,obj.settings.c,obj.settings.e]
            bounds_max = [obj.settings.b,obj.settings.d,obj.settings.f]

            # tracer energy settings
            nE = 128
            wE = 1.0
            E_min = 0.0011
            phiTracer = zeros(prod(Ncells),nE,nB)

            #set up material parameters
            root = pkgdir(TITUS) 
            tables = load_proton_tables(
                joinpath(root, "src/raytracer/data/proton_totalXS_data"),
                joinpath(root, "src/raytracer/data/proton_S_data_topas")
            )
            
            #create array with just HU indices
            unique_vals = sort(Base.unique(Int64.(obj.densityVecHU)))
            val_to_idx = Dict(val => i for (i,val) in enumerate(unique_vals))
            CT_idx = map(x -> val_to_idx[x], obj.densityVecHU)

            xs_db = build_xs_database(
                tables,
                unique_vals)
            
            if length(coords_bounds)!=0
                phiTracer = zeros(size(coords_bounds,2),nE,nB)
            end

            for b=1:nB
                mu_E = Float64(obj.settings.mu_e[b])
                sigma_E = 0.01 * mu_E
                E_max = mu_E + 7*sigma_E

                #choose mode of ray assembly depending on whether direction is close to being aligned with coordinate axis
                if length(coords_bounds) == 0 #no cell coordinates given -> not in octree trace mode
                    if  maximum(abs.(tracerDirs[b,:] / norm(tracerDirs[b,:]))) ≥ 0.999 #this tolerance can be adjusted 
                        trace_mode = :incoming_plane
                    else
                        trace_mode = :midpoints
                    end
                    println("Using mode $trace_mode for tracing")
                    # get ray positions and weights
                    x_start, x_end = get_rayPositions(tracerDirs[b,:],bounds_min,bounds_max,Ncells,trace_mode,zeros(0,0))
                else
                    trace_mode = :octree
                    println("Using mode $trace_mode for tracing")
                    # get ray positions and weights
                    x_start, x_end = get_rayPositions(tracerDirs[b,:],bounds_min,bounds_max,Ncells,trace_mode,coords_bounds)
                end

                # determine weight according to beam distribution 
                # pos_xyz = rotate_bev_to_xyz(tracerDirs[b,:])*x_start'
                pos_xyz = rotate_xyz_to_bev(tracerDirs[b,:])*x_start'
                x0 = rotate_xyz_to_bev(tracerDirs[b,:])*[obj.settings.x0[b],obj.settings.y0[b],obj.settings.z0[b]]
                μs = [x0[1],x0[2]]; σs = [obj.settings.sigmaX,obj.settings.sigmaY]; 
                pdf_rays = Product([Normal(μ,σ) for (μ,σ) in zip(μs,σs)])
                pos_w = pdf(pdf_rays,pos_xyz[1:2,:])

                println("trace_mode = $trace_mode")
                #discard rays too far from beam center
                if trace_mode != :octree
                    println("Reducing number of rays")
                    valid_indices = [i for i in 1:size(pos_xyz, 2) if 
                    μs[1] .- 2*σs[1] ≤ pos_xyz[1, i] ≤ μs[1] .+ 2*σs[1] &&
                    μs[2] .- 2*σs[2] ≤ pos_xyz[2, i] ≤ μs[2] .+ 2*σs[2] ]#&&
                    #pos_end[3,i] ≤ 1.2*0.0022*(obj.settings.eMax-obj.settings.eRest)^1.77] # also include something here to filter out cells clearly behind range
                    x_start = x_start[valid_indices,:]
                    x_end = x_end[valid_indices,:]
                    pos_w = pos_w[valid_indices]
                end
                numRays = size(x_start,1)
                
                E_bounds, E_mid, dE = setup_RTEnergyGrps(nE,E_min,E_max;spacing=:uniformSafe)   
                E_tracer = E_mid .+ obj.settings.eRest #this needs to be group mids and total (not kinetic) energy

                S_star,T_star,Sigma_abs = get_material_xs(xs_db, unique_vals, E_bounds)
                if obj.settings.solverName == "Split"
                    Sigma_abs .= 0.0;
                end
                println("Starting to trace $numRays rays")
                if numRays == 0
                    println("Warning: Tracing 0 rays! Check whether beam mean position $x0 is too close to boundaries or rays are wrongly discarded.")
                end

                #run tracer
                if trace_mode == :incoming_plane
                    phi_lock = SpinLock()
                    Threads.@threads for r in 1:numRays
                        touched_ids, touched_vals = calc_uncollided_flux(
                            x_start[r, :], x_end[r, :],
                            bounds_min, bounds_max, Ncells,
                            Sigma_abs, S_star, T_star,
                            CT_idx[:], mu_E, sigma_E,
                            E_bounds, wE, nE
                        )
                        w = pos_w[r]
                        lock(phi_lock) do
                            for i in eachindex(touched_ids)
                                @views phiTracer[touched_ids[i], :, b] .+= w .* touched_vals[i, :]
                            end
                        end
                    end
                elseif trace_mode == :midpoints
                    phi_lock = SpinLock()
                    Threads.@threads for r in 1:numRays
                        touched_ids, touched_vals = calc_uncollided_flux(
                            x_start[r, :], x_end[r, :],
                            bounds_min, bounds_max, Ncells,
                            Sigma_abs, S_star, T_star,
                            CT_idx[:], mu_E, sigma_E,
                            E_bounds, wE, nE
                        )
                        target_cell = valid_indices[r]
                        w = pos_w[r]
                        lock(phi_lock) do
                            @views phiTracer[target_cell, :, b] .+= w .* touched_vals[end, :]
                        end
                    end
                else 
                    println("No valid tracing mode given, choose :incoming_plane, :midpoints or :octree.")
                end
                phiTracer[:,:,b] = phiTracer[:,:,b]./sum(phiTracer[:,:,b]) * mu_E;
  
            end 
        else #read from file (results of fortran tracer)

            root = pkgdir(TITUS) 
            file_path = joinpath(root, "tracer_results/$(obj.settings.tracerFileName).bin")
            if isfile(file_path)
                phiTracer = Array{Float64}(undef,Int.(stat(file_path).size/8))
                io_phi = open(file_path, "r")
            else
                error("no file $(file_path) detected");
            end

            # set up energy grid
            nE = 128
            E_bounds, E_mid, dE = setup_RTEnergyGrps(nE-1,0.011,obj.settings.eMax .- obj.settings.eRest);
            E_tracer = E_bounds .+ obj.settings.eRest 
            E_tracer[1] = E_tracer[1].-0.001 # relevant for Fortran tracer results
            read!(io_phi,phiTracer);
            close(io_phi)
            phiTracer = reshape(phiTracer,(obj.settings.sizeOfTracerCT[1]*obj.settings.sizeOfTracerCT[2]*obj.settings.sizeOfTracerCT[3],nE,:))
        end
    end
    return E_tracer[end:-1:1], phiTracer[:,end:-1:1,1:nB]
end

function truncateToFixedRankCUDA(obj::SolverGPU{T},X::CuArray{T,2},S::CuArray{T,2},W::CuArray{T,2}) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(Matrix(S));
    rmax = obj.settings.rMax;
    Utilde = CuArray(U[:, 1:rmax])
    Vtilde = CuArray(V[:, 1:rmax])

    # return rank
    return X*Utilde, CuArray(diagm(D[1:rmax])), W*Vtilde;
end


function truncateCUDANew(obj::SolverGPU{T},
                      X::CuArray{T,2},
                      S::CuArray{T,2},
                      W::CuArray{T,2}) where {T<:AbstractFloat}

    rMaxTotal = obj.settings.rMax
    rMinTotal = 2

    U, D, V = svd(S)          # CUSOLVER path; D is a CuVector, U/V are CuArrays

    D_cpu   = Vector(D)
    n       = length(D_cpu)
    tail_sq = sum(abs2, D_cpu)
    tol     = T(obj.settings.epsAdapt) * tail_sq^(T(obj.settings.adaptIndex)/2)
    rmax    = n

    for j = 1:n
        tail_sq -= D_cpu[j]^2
        if tail_sq < tol^2          # compare squared to avoid sqrt each iteration
            rmax = j
            break
        end
    end

    rmax = clamp(rmax, rMinTotal, rMaxTotal)

    # Slice and project — U[:,1:rmax] and V[:,1:rmax] are already on GPU
    Utilde = U[:, 1:rmax]
    Vtilde = V[:, 1:rmax]

    # Fuse diagonal scaling into S rather than forming a full diagm matrix
    Snew = CuArray(Diagonal(D_cpu[1:rmax]))

    return X * Utilde, Snew, W * Vtilde
end
