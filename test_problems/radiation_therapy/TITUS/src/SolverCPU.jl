__precompile__

using ProgressMeter
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum
using Base.Threads
using Interpolations
using TimerOutputs
using Random, Distributions

include("stencilsCPU.jl")

mutable struct SolverCPU{T<:AbstractFloat}
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
    pn::PNSystemCPU;

    # stencil matrices
    stencil::UpwindStencil3D;

    # material density
    density::Array{T,3};
    densityVec::Array{T,1};

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
    function SolverCPU(settings,order=2)
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

        # construct CSD fields
        csd = CSD(settings,T);

        # set density vector
        density = T.(settings.density);

        # allocate dose vector
        dose = zeros(T,nx*ny*nz)
        pn = PNSystemCPU(settings,T)
        SetupSystemMatricesSparse(pn)
        stencil = UpwindStencil3D(settings,order);
        Norder = (settings.nPN+1)^2

        # collect boundary indices
        if order == 1
            boundaryIdx = zeros(Int,2*nx*ny+2*ny*nz + 2*nx*nz)
            Threads.@threads for i = 1:nx
                Threads.@threads for k = 1:nz
                    j = 1;
                    boundaryIdx[(i-1)*nz+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                    j = ny;
                    boundaryIdx[(i-1)*nz+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                end
            end
            Threads.@threads for i = 1:nx
                Threads.@threads for j = 1:ny
                    k = 1;
                    boundaryIdx[2*nx*nz+(i-1)*ny+(j-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                    k = nz;
                    boundaryIdx[2*nx*nz+(i-1)*ny+(j-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                end
            end
    
            Threads.@threads for j = 1:ny
                Threads.@threads for k = 1:nz
                    i = 1;
                    boundaryIdx[2*nx*ny+2*nx*nz+(j-1)*nz+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                    i = nx;
                    boundaryIdx[2*nx*ny+2*nx*nz+(j-1)*nz+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
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

        # setup quadrature
        qorder = 1 
        if iseven(qorder) qorder += 1; end 
        qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
        Q = Quadrature(qorder,qtype);

        O,M = ComputeTrafoMatrices(Q,Norder,settings.nPN,[settings.Omega1 settings.Omega2 settings.Omega3]);
        densityVec = Ten2Vec(density);

        new{T}(T.(x),T.(y),T.(z),order,settings,gamma,csd,pn,stencil,density,densityVec,dose,boundaryIdx,Q,T.(O),T.(M),T);
    end
end

function SolveTracer_fullPnInEnergy(obj::SolverCPU{T}, model::String="Boltzmann", trace::Bool=false) where {T<:AbstractFloat}
    order = obj.order

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    s = obj.settings;

    # Run raytracer or load solution
    @timeit to "Ray-tracer" begin
        # determine uncollided flux with tracer   
        E_tracer, psiE = RunTracer_UniDirectional(obj,model,trace) 
    end
    
    q = [obj.settings.Omega1; obj.settings.Omega2; obj.settings.Omega3]
    normQ = zeros(size(obj.Q.pointsxyz,1))
    for k = 1:size(obj.Q.pointsxyz,1)
        normQ[k] = norm(obj.Q.pointsxyz[k,:] .- q)
    end
    _,idxBeam = findmin(normQ)
    #idxBeam = 1;
    
    qReduced = T.(obj.Q.pointsxyz[idxBeam,:])
    MReduced = T.(obj.M[:,idxBeam])
    OReduced = T.(obj.O[idxBeam,:])
    nq = length(idxBeam);

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.densityVec);
    Id = Diagonal(ones(T,N));

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE
    densityVec = Array(obj.densityVec)

    Id = Diagonal(ones(T,N));
    idx = Base.unique(i -> obj.densityVec[i], 1:length(obj.densityVec))
    idxK = Vector{Vector{Int64}}([])
    Threads.@threads for k=1:length(idx)
    push!(idxK,findall(i->(i==obj.densityVec[idx[k]]),obj.densityVec))
    end

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy,obj.settings.dz)*maximum(densityInv))

    prog = Progress(nEnergies-1,1)
    t = 0;

    u = zeros(T,nx*ny*nz,N);
    dose = zeros(T,nx*ny*nz);
    dose_coll = zeros(T,nx*ny*nz);
    psi = zeros(T,nx*ny*nz,1);

    stencil = UpwindStencil3D(obj.settings, order)

    D⁺₁ = stencil.D⁺₁
    D⁺₂ = stencil.D⁺₂
    D⁺₃ = stencil.D⁺₃
    D⁻₁ = stencil.D⁻₁
    D⁻₂ = stencil.D⁻₂
    D⁻₃ = stencil.D⁻₃

    Ax,Ay,Az = SetupSystemMatrices(obj.pn);
    Σ₁ = eigvals(Ax)
    T₁ = eigvecs(Ax)
    T₁⁻¹ = T₁'
    Σ₁⁺ = copy(Σ₁); Σ₁⁺[Σ₁⁺ .< 0] .= 0;
    Σ₁⁻ = copy(Σ₁); Σ₁⁻[Σ₁⁻ .> 0] .= 0;

    Σ₂ = eigvals(Ay)
    T₂ = eigvecs(Ay)
    T₂⁻¹ = T₂'
    Σ₂⁺ = copy(Σ₂); Σ₂⁺[Σ₂⁺ .< 0] .= 0;
    Σ₂⁻ = copy(Σ₂); Σ₂⁻[Σ₂⁻ .> 0] .= 0;

    Σ₃ = eigvals(Az)
    T₃ = eigvecs(Az)
    T₃⁻¹ = T₃'
    Σ₃⁺ = copy(Σ₃); Σ₃⁺[Σ₃⁺ .< 0] .= 0;
    Σ₃⁻ = copy(Σ₃); Σ₃⁻[Σ₃⁻ .> 0] .= 0;
    ∫Y₀⁰dΩ = T(4 * pi / sqrt(4 * pi)); 

    @timeit to "Interpolate to energy grid" begin
        nPsi = size(psiE,1)
        nB = size(psiE,3)
        psi = zeros(T,nx*ny*nz,nB);
        E_tracer[1] = E_tracer[1].-0.001
        if nB == 1
            ETracer2E = interpolate((1:nPsi,E_tracer[1:end]), psiE[:,:,1],(NoInterp(),Gridded(Linear())))
        else
            ETracer2E = interpolate((1:nPsi,E_tracer[1:end],1:nB), psiE,(NoInterp(),Gridded(Linear()),NoInterp()))
        end
        psiE = nothing
    end

    Sinv = zeros(T,nPsi)
    
    wMat = T.(matComp(obj.settings.densityHU[:]).*obj.settings.density[:]'./100);
    Nmat = size(wMat,1)
    #loop over energy
    @timeit to "Pn" begin
        for n=2:nEnergies
            dE = energy[n-1] - energy[n]
             dEGrid = energy[n-1] - energy[n]
             # compute scattering coefficients at current energy
             if model == "FP" 
                sigmaS = zeros(1,Nmat); #FP, outscattering included in coefficient xi
                xi = T.(XiAtEnergyandX(obj.csd,energy[n])) #transport crosssection FP
             else 
                sigmaS = SigmaAtEnergyandX(obj.csd,energy[n])
             end

             DvecCPU = zeros(obj.pn.nTotalEntries,Nmat)
             if model == "FP"
                for j=1:Nmat
                    for l = 0:obj.pn.N
                         for k=-l:l
                             i = GlobalIndex( l, k );
                            DvecCPU[i+1,j] = 0.5*xi[1,j]*(-l*(l+1))#Fokker-Planck
                        end
                     end
                 end
             else
                for j=1:Nmat
                    for l = 0:obj.pn.N
                        for k=-l:l
                            i = GlobalIndex( l, k );
                            DvecCPU[i+1,j] = sigmaS[l+1,j] #Boltzmann
                        end
                    end
                end
             end

            stoppingPowerCurrent = get_S_at_energy(obj.csd, energy[n-1], obj.settings.density[:])
            for j=1:length(idx)
                Sinv[idxK[j]] .= 1 ./stoppingPowerCurrent[j]
            end

             if nB == 1 
                psi .= ETracer2E.(1:nPsi,obj.csd.eGrid[n]) #stopping power already included in tracer
             else
                for b=1:nB
                    psi[:,b] .= ETracer2E.(1:nPsi,obj.csd.eGrid[n],b)#stopping power already included in tracer
                end
             end
            if n > 2 # perform streaming update after first collision (before solution is zero)
                function Fx(t,v)
                    return - (D⁺₁*(Sinv.*v)*Diagonal(Σ₁⁺) .+ D⁻₁*(Sinv.*v)*Diagonal(Σ₁⁻))
                end
            
                function Fy(t,v)
                    return - (D⁺₂*(Sinv.*v)*Diagonal(Σ₂⁺) .+ D⁻₂*(Sinv.*v)*Diagonal(Σ₂⁻))
                end
                function Fz(t,v)
                    return - (D⁺₃*(Sinv.*v)*Diagonal(Σ₃⁺) .+ D⁻₃*(Sinv.*v)*Diagonal(Σ₃⁻))
                end
                v = u*T₁;
                u .= rk4(dE, Fx, v) * T₁⁻¹
                v .= u*T₂;
                u .= rk4(dE, Fy, v) * T₂⁻¹
                v .= u*T₃;
                u .= rk4(dE, Fz, v) * T₃⁻¹
                u[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero
            end
            v = u
            #in- and outscattering collided 
            for j=1:Nmat
                #implicit_update!(u,T.((DvecCPU[:,j].-sigmaS[1,j])),wMat[j,:].*Sinv,dE)
                u .+= dE*((DvecCPU[:,j].-sigmaS[1,j])'.*v.*wMat[j,:].*Sinv).+ dE* DvecCPU[:,j]'.*(psi * MReduced') .*wMat[j,:].*Sinv;
            end

            u[obj.boundaryIdx,:] .= 0.0;
            ############## Dose Computation ##############
            dose .+= dEGrid * (u[:,1]+psi*MReduced[1,:])* ∫Y₀⁰dΩ  
            dose_coll .+= dEGrid *(u[:,1]* ∫Y₀⁰dΩ) 
            
            next!(prog) # update progress bar
        end
    end

    # return solution and dose
    return dose,dose_coll
end

function SolveTracer_fullPnInEnergy_FP(obj::SolverCPU{T}, model::String="Boltzmann", trace::Bool=false) where {T<:AbstractFloat}
    order = obj.order

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    s = obj.settings;

    # Run raytracer or load solution
    @timeit to "Ray-tracer" begin
        # determine uncollided flux with tracer   
        E_tracer, psiE = RunTracer_UniDirectional(obj,model,trace) 
    end
    
    q = [obj.settings.Omega1; obj.settings.Omega2; obj.settings.Omega3]
    normQ = zeros(size(obj.Q.pointsxyz,1))
    for k = 1:size(obj.Q.pointsxyz,1)
        normQ[k] = norm(obj.Q.pointsxyz[k,:] .- q)
    end
    _,idxBeam = findmin(normQ)
    #idxBeam = 1;
    
    qReduced = T.(obj.Q.pointsxyz[idxBeam,:])
    MReduced = T.(obj.M[:,idxBeam])
    OReduced = T.(obj.O[idxBeam,:])
    nq = length(idxBeam);

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.densityVec);
    Id = Diagonal(ones(T,N));

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE
    densityVec = Array(obj.densityVec)

    Id = Diagonal(ones(T,N));
    idx = Base.unique(i -> obj.densityVec[i], 1:length(obj.densityVec))
    idxK = Vector{Vector{Int64}}([])
    Threads.@threads for k=1:length(idx)
    push!(idxK,findall(i->(i==obj.densityVec[idx[k]]),obj.densityVec))
    end

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy,obj.settings.dz)*maximum(densityInv))

    prog = Progress(nEnergies-1,1)
    t = 0;

    u = zeros(T,nx*ny*nz,N);
    dose = zeros(T,nx*ny*nz);
    dose_coll = zeros(T,nx*ny*nz);
    psi = zeros(T,nx*ny*nz,1);

    stencil = UpwindStencil3D(obj.settings, order)

    D⁺₁ = stencil.D⁺₁
    D⁺₂ = stencil.D⁺₂
    D⁺₃ = stencil.D⁺₃
    D⁻₁ = stencil.D⁻₁
    D⁻₂ = stencil.D⁻₂
    D⁻₃ = stencil.D⁻₃

    Ax,Ay,Az = SetupSystemMatrices(obj.pn);
    Σ₁ = eigvals(Ax)
    T₁ = eigvecs(Ax)
    T₁⁻¹ = T₁'
    Σ₁⁺ = copy(Σ₁); Σ₁⁺[Σ₁⁺ .< 0] .= 0;
    Σ₁⁻ = copy(Σ₁); Σ₁⁻[Σ₁⁻ .> 0] .= 0;

    Σ₂ = eigvals(Ay)
    T₂ = eigvecs(Ay)
    T₂⁻¹ = T₂'
    Σ₂⁺ = copy(Σ₂); Σ₂⁺[Σ₂⁺ .< 0] .= 0;
    Σ₂⁻ = copy(Σ₂); Σ₂⁻[Σ₂⁻ .> 0] .= 0;

    Σ₃ = eigvals(Az)
    T₃ = eigvecs(Az)
    T₃⁻¹ = T₃'
    Σ₃⁺ = copy(Σ₃); Σ₃⁺[Σ₃⁺ .< 0] .= 0;
    Σ₃⁻ = copy(Σ₃); Σ₃⁻[Σ₃⁻ .> 0] .= 0;
    ∫Y₀⁰dΩ = T(4 * pi / sqrt(4 * pi)); 

    @timeit to "Interpolate to energy grid" begin
        nPsi = size(psiE,1)
        nB = size(psiE,3)
        psi = zeros(T,nx*ny*nz,nB);
        E_tracer[1] = E_tracer[1].-0.001
        if nB == 1
            ETracer2E = interpolate((1:nPsi,E_tracer[1:end]), psiE[:,:,1],(NoInterp(),Gridded(Linear())))
        else
            ETracer2E = interpolate((1:nPsi,E_tracer[1:end],1:nB), psiE,(NoInterp(),Gridded(Linear()),NoInterp()))
        end
        psiE = nothing
    end

    Sinv = zeros(T,nPsi)
    
    wMat = T.(matComp(obj.settings.densityHU[:]).*obj.settings.density[:]'./100);
    Nmat = size(wMat,1)
    #loop over energy
    @timeit to "Pn" begin
        for n=2:nEnergies
            dE = energy[n-1] - energy[n]
             dEGrid = energy[n-1] - energy[n]
             # compute scattering coefficients at current energy
             if model == "FP" 
                sigmaS = zeros(1,Nmat); #FP, outscattering included in coefficient xi
                xi = T.(XiAtEnergyandX(obj.csd,energy[n])) #transport crosssection FP
             else 
                sigmaS = SigmaAtEnergyandX(obj.csd,energy[n])
             end

             DvecCPU = zeros(obj.pn.nTotalEntries,Nmat)
             if model == "FP"
                 N_corr = 19;
                for j=1:Nmat
                    for l = 0:obj.pn.N
                         for k=-l:l
                             i = GlobalIndex( l, k );
                            #DvecCPU[i+1,j] = 0.5*xi[1,j]*(-l*(l+1))#Fokker-Planck
                             DvecCPU[i+1,j] = 0.5*xi[1,j]*(N_corr*(N_corr+1)-l*(l+1)) #with correction
                        end
                     end
                    sigmaS[1,j] = 0.5*xi[1,j]*(N_corr*(N_corr+1))
                 end
             else
                for j=1:Nmat
                    for l = 0:obj.pn.N
                        for k=-l:l
                            i = GlobalIndex( l, k );
                            DvecCPU[i+1,j] = sigmaS[l+1,j] #Boltzmann
                        end
                    end
                end
             end

            stoppingPowerCurrent = get_S_at_energy(obj.csd, energy[n-1], obj.settings.density[:])
            for j=1:length(idx)
                Sinv[idxK[j]] .= 1 ./ stoppingPowerCurrent[j]
             end

             if nB == 1 
                psi .= ETracer2E.(1:nPsi,obj.csd.eGrid[n]) #stopping power already included in tracer
             else
                for b=1:nB
                    psi[:,b] .= ETracer2E.(1:nPsi,obj.csd.eGrid[n],b)#stopping power already included in tracer
                end
             end
            if n > 2 # perform streaming update after first collision (before solution is zero)
                function Fx(t,v)
                    return - (D⁺₁*(Sinv.*v)*Diagonal(Σ₁⁺) .+ D⁻₁*(Sinv.*v)*Diagonal(Σ₁⁻))
                end
            
                function Fy(t,v)
                    return - (D⁺₂*(Sinv.*v)*Diagonal(Σ₂⁺) .+ D⁻₂*(Sinv.*v)*Diagonal(Σ₂⁻))
                end
                function Fz(t,v)
                    return - (D⁺₃*(Sinv.*v)*Diagonal(Σ₃⁺) .+ D⁻₃*(Sinv.*v)*Diagonal(Σ₃⁻))
                end
                v = u*T₁;
                u .= rk4(dE, Fx, v) * T₁⁻¹
                v .= u*T₂;
                u .= rk4(dE, Fy, v) * T₂⁻¹
                v .= u*T₃;
                u .= rk4(dE, Fz, v) * T₃⁻¹
                u[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero
            end
            v = u
            #in- and outscattering collided 
            for j=1:Nmat
                #implicit_update!(u,T.((DvecCPU[:,j].-sigmaS[1,j])),wMat[j,:].*Sinv,dE)
                u .+= dE*((DvecCPU[:,j].-sigmaS[1,j])'.*v.*wMat[j,:].*Sinv).+ dE* DvecCPU[:,j]'.*(psi * MReduced') .*wMat[j,:].*Sinv;
            end

            u[obj.boundaryIdx,:] .= 0.0;
            ############## Dose Computation ##############
            dose .+= dEGrid * (u[:,1]+psi*MReduced[1,:])* ∫Y₀⁰dΩ  
            dose_coll .+= dEGrid *(u[:,1]* ∫Y₀⁰dΩ) 
            
            next!(prog) # update progress bar
        end
    end

    # return solution and dose
    return dose,dose_coll
end

function implicit_update!(
    L::AbstractMatrix{T},
    d::AbstractVector{T},
    b::AbstractVector{T},
    dt::T
) where {T<:AbstractFloat}

    n, m = size(L)
    @assert length(d) == m
    @assert length(b) == n

    @inbounds for i in 1:m, j in 1:n
        L[j, i] /= (1 - dt * d[i] * b[j])
    end

    return L
end

function implicit_L_step!(L::Matrix{T},
                           d::Vector{T},
                           B::Matrix{T},
                           dt::T) where {T<:AbstractFloat}
    m, r = size(L)
    @assert length(d) == m

    Bt = transpose(B)

    # Factorize (I - dt * dj * Bᵀ) once per unique value of d, then reuse
    # the factors for all rows that share that value. This matches the GPU
    # version's grouping strategy and avoids redundant factorisations.
    unique_d  = Base.unique(d)
    d_to_factor = Dict{T, LU{T, Matrix{T}, Vector{Int}}}()
    for dj in unique_d
        A = Matrix{T}(I, r, r) .- dt * dj * Bt
        d_to_factor[dj] = lu(A)
    end

    groups = Dict{T, Vector{Int}}()
    for i in 1:m
        push!(get!(groups, d[i], Int[]), i)
    end

    # Solve (I - dt * dj * Bᵀ)ᵀ * L[i,:]ᵀ = L_old[i,:]ᵀ for each group.
    # The system is (I - dt*dj*Bᵀ) x = rhs, solved via the LU factors.
    # Each row of L is treated as a right-hand side column vector.
    for (dj, row_idx) in groups
        F    = d_to_factor[dj]
        rows = @view L[row_idx, :]        # (n_rows × r)
        rows_t = copy(transpose(rows))    # (r × n_rows) — needed for ldiv!
        ldiv!(F, rows_t)                  # solve in-place: F \ rows_t
        transpose!(rows, rows_t)          # write result back into L
    end

    return L
end

function Solve(obj::SolverCPU{T}, model::String="Boltzmann",
                                           trace::Bool=false,
                                           model_collided::String="Boltzmann") where {T<:AbstractFloat}
    r     = Int(floor(obj.settings.r / 2));
    order = obj.order

    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N  = obj.pn.nTotalEntries;
    s  = obj.settings;

    @timeit to "Ray-tracer" begin
        E_tracer, psiE = RunTracer_UniDirectional(obj, model, trace)
    end

    q = [obj.settings.Omega1; obj.settings.Omega2; obj.settings.Omega3]
    normQ = zeros(size(obj.Q.pointsxyz, 1))
    for k = 1:size(obj.Q.pointsxyz, 1)
        normQ[k] = norm(obj.Q.pointsxyz[k,:] .- q)
    end
    _, idxBeam = findmin(normQ)

    qReduced = T.(obj.Q.pointsxyz[idxBeam,:])
    MReduced = T.(obj.M[:,idxBeam])
    M1       = MReduced[1,:]
    OReduced = T.(obj.O[idxBeam,:])
    nq       = length(idxBeam);

    densityInv = Diagonal(1.0 ./ obj.densityVec);
    Id         = Diagonal(ones(T, N));

    X, _, _ = svd(zeros(T, nx*ny*nz, r));
    W, _, _ = svd(zeros(T, N, r));

    X = Matrix(X[:,1:r]);
    W = Matrix(W[:,1:r]);
    S = zeros(T, r, r);
    K = zeros(T, size(X));

    MUp = zeros(T, r, r)
    NUp = zeros(T, r, r)

    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies       = length(energy);
    dE              = energy[1] - energy[2]
    obj.settings.dE = dE
    densityVec      = obj.densityVec

    Id   = Diagonal(ones(T, N));
    idx  = Base.unique(i -> densityVec[i], 1:length(densityVec))
    idxK = Vector{Vector{Int64}}([])
    Threads.@threads for k = 1:length(idx)
        push!(idxK, findall(i -> (i == densityVec[idx[k]]), densityVec))
    end

    # The Boltzmann variant includes densityInv in the CFL print; the FP variant does not.
    if model_collided == "Boltzmann"
        println("CFL = ", dE / min(obj.settings.dx, obj.settings.dy, obj.settings.dz) * maximum(densityInv))
    else
        println("CFL = ", dE / min(obj.settings.dx, obj.settings.dy, obj.settings.dz))
    end

    prog = Progress(nEnergies-1, 1)
    rVec = r .* ones(2, nEnergies)

    uOUnc     = zeros(T, nx*ny*nz);
    dose_coll = zeros(T, nx*ny*nz);
    dose      = zeros(T, nx*ny*nz);
    e1        = zeros(T, N); e1[1] = 1.0;

    @timeit to "Setup upwind stencil" begin
        stencil = UpwindStencil3D(obj.settings, order)

        D⁺₁ = stencil.D⁺₁
        D⁺₂ = stencil.D⁺₂
        D⁺₃ = stencil.D⁺₃
        D⁻₁ = stencil.D⁻₁
        D⁻₂ = stencil.D⁻₂
        D⁻₃ = stencil.D⁻₃

        Ax, Ay, Az = SetupSystemMatrices(obj.pn);
        Σ₁  = eigvals(Ax); T₁  = eigvecs(Ax); T₁⁻¹ = T₁'
        Σ₁⁺ = copy(Σ₁); Σ₁⁺[Σ₁⁺ .< 0] .= 0;
        Σ₁⁻ = copy(Σ₁); Σ₁⁻[Σ₁⁻ .> 0] .= 0;

        Σ₂  = eigvals(Ay); T₂  = eigvecs(Ay); T₂⁻¹ = T₂'
        Σ₂⁺ = copy(Σ₂); Σ₂⁺[Σ₂⁺ .< 0] .= 0;
        Σ₂⁻ = copy(Σ₂); Σ₂⁻[Σ₂⁻ .> 0] .= 0;

        Σ₃  = eigvals(Az); T₃  = eigvecs(Az); T₃⁻¹ = T₃'
        Σ₃⁺ = copy(Σ₃); Σ₃⁺[Σ₃⁺ .< 0] .= 0;
        Σ₃⁻ = copy(Σ₃); Σ₃⁻[Σ₃⁻ .> 0] .= 0;

        ∫Y₀⁰dΩ = T(4 * pi / sqrt(4 * pi));
    end

    @timeit to "Interpolate to energy grid" begin
        nPsi = size(psiE, 1)
        nB   = size(psiE, 3)
        psi  = zeros(T, nx*ny*nz, nB);
        E_tracer[1] = E_tracer[1] .- 0.001
        if nB == 1
            ETracer2E = interpolate((1:nPsi, E_tracer[1:end]), psiE[:,:,1],
                                    (NoInterp(), Gridded(Linear())))
        else
            ETracer2E = interpolate((1:nPsi, E_tracer[1:end], 1:nB), psiE,
                                    (NoInterp(), Gridded(Linear()), NoInterp()))
        end
        psiE = nothing
    end

    Sinv = zeros(T, nPsi)
    wMat = T.(matComp(obj.settings.densityHU[:]) .* obj.settings.density[:]' ./ 100);
    Nmat = size(wMat, 1)

    println("Starting energy loop")
    @timeit to "DLRA" begin
        for n = 2:nEnergies
            dE     = energy[n-1] - energy[n]
            dEGrid = energy[n-1] - energy[n]

            # model_collided selects the scattering operator.
            # FP: transport cross-section xi; sigmaS is zero and populated below.
            # Boltzmann: sigmaS read directly from the CSD tables.
            if model_collided == "FP"
                xi     = T.(XiAtEnergyandX(obj.csd, energy[n]))
                sigmaS = zeros(1, Nmat);
            else
                sigmaS = SigmaAtEnergyandX(obj.csd, energy[n])
            end

            Dvec = zeros(obj.pn.nTotalEntries, Nmat)
            if model_collided == "FP"
                N_corr = 19;
                for j = 1:Nmat
                    for l = 0:obj.pn.N
                        for k = -l:l
                            i = GlobalIndex(l, k);
                            Dvec[i+1, j] = 0.5 * xi[1,j] * (N_corr*(N_corr+1) - l*(l+1))
                        end
                    end
                    sigmaS[1,j] = 0.5 * xi[1,j] * (N_corr*(N_corr+1))
                end
            else
                for j = 1:Nmat
                    for l = 0:obj.pn.N
                        for k = -l:l
                            i = GlobalIndex(l, k);
                            Dvec[i+1, j] = sigmaS[l+1, j]
                        end
                    end
                end
            end
            sigmaS1 = T.(sigmaS[1,:])

            stoppingPowerCurrent = get_S_at_energy(obj.csd, energy[n-1], obj.settings.density[:])
            for j=1:length(idx)
                Sinv[idxK[j]] .= 1 ./stoppingPowerCurrent[j]
            end

            if nB == 1
                psi .= ETracer2E.(1:nPsi, obj.csd.eGrid[n])
            else
                for b = 1:nB
                    psi[:,b] .= ETracer2E.(1:nPsi, obj.csd.eGrid[n], b)
                end
            end

            if n > 2
                function FLx(t, L)
                    return -(Σ₁⁺.*L*(X'*D⁺₁*(Sinv.*X))' .+ Σ₁⁻.*L*(X'*D⁻₁*(Sinv.*X))')
                end
                function FLy(t, L)
                    return -(Σ₂⁺.*L*(X'*D⁺₂*(Sinv.*X))' .+ Σ₂⁻.*L*(X'*D⁻₂*(Sinv.*X))')
                end
                function FLz(t, L)
                    return -(Σ₃⁺.*L*(X'*D⁺₃*(Sinv.*X))' .+ Σ₃⁻.*L*(X'*D⁻₃*(Sinv.*X))')
                end

                function FKx(t, K)
                    return -(D⁺₁*(Sinv.*K)*((W'*T₁)*(Σ₁⁺.*(T₁⁻¹*W))) .+ D⁻₁*(Sinv.*K)*((W'*T₁)*(Σ₁⁻.*(T₁⁻¹*W))))
                end
                function FKy(t, K)
                    return -(D⁺₂*(Sinv.*K)*((W'*T₂)*(Σ₂⁺.*(T₂⁻¹*W))) .+ D⁻₂*(Sinv.*K)*((W'*T₂)*(Σ₂⁻.*(T₂⁻¹*W))))
                end
                function FKz(t, K)
                    return -(D⁺₃*(Sinv.*K)*((W'*T₃)*(Σ₃⁺.*(T₃⁻¹*W))) .+ D⁻₃*(Sinv.*K)*((W'*T₃)*(Σ₃⁻.*(T₃⁻¹*W))))
                end

                ################## K-step ##################
                X[obj.boundaryIdx,:] .= 0.0;
                K  = X * S;
                K .= rk4(dE, FKx, K)
                K .= rk4(dE, FKy, K)
                K .= rk4(dE, FKz, K)
                Xtmp, _, _ = svd([X K]); MUp = Xtmp' * X;

                ################## L-step ##################
                L  = T₁⁻¹ * W * S';
                L .= T₁ * rk4(dE, FLx, L)
                L .= T₂⁻¹ * L
                L .= T₂ * rk4(dE, FLy, L)
                L .= T₃⁻¹ * L
                L .= T₃ * rk4(dE, FLz, L)

                Wtmp, _, _ = svd([W L]); NUp = Wtmp' * W;
                X = Xtmp;
                W = Wtmp;
                X[obj.boundaryIdx,:] .= 0.0;

                ################## S-step ##################
                function FSx(t, S)
                    return -((X'*D⁺₁*(Sinv.*X))*S*((W'*T₁)*(Σ₁⁺.*(T₁⁻¹*W))) .+ (X'*D⁻₁*(Sinv.*X))*S*((W'*T₁)*(Σ₁⁻.*(T₁⁻¹*W))))
                end
                function FSy(t, S)
                    return -((X'*D⁺₂*(Sinv.*X))*S*((W'*T₂)*(Σ₂⁺.*(T₂⁻¹*W))) .+ (X'*D⁻₂*(Sinv.*X))*S*((W'*T₂)*(Σ₂⁻.*(T₂⁻¹*W))))
                end
                function FSz(t, S)
                    return -((X'*D⁺₃*(Sinv.*X))*S*((W'*T₃)*(Σ₃⁺.*(T₃⁻¹*W))) .+ (X'*D⁻₃*(Sinv.*X))*S*((W'*T₃)*(Σ₃⁻.*(T₃⁻¹*W))))
                end
                S  = MUp * S * (NUp')
                S .= rk4(dE, FSx, S)
                S .= rk4(dE, FSy, S)
                S .= rk4(dE, FSz, S)

                X, S, W = truncate!(obj, T.(X), T.(S), T.(W));
                r = size(S, 1)
            end

            ############# Out-scattering ##############
            L  = W * S';
            L0 = L
            for j = 1:Nmat
                L += dE * (Dvec[:,j] .- sigmaS[1,j]) .* (L0 * X' * (wMat[j,:] .* Sinv .* X));
            end
            W, S1, S2 = svd(L)
            S .= S2 * Diagonal(S1)

            ############## In-scattering ##############
            X[obj.boundaryIdx,:] .= 0.0;
            K = X * S;
            for j = 1:Nmat
                K += dE * wMat[j,:] .* Sinv .* psi * MReduced' * (Dvec[:,j] .* W);
            end
            K[obj.boundaryIdx,:] .= 0.0;

            Xtmp, _, _ = svd([X K]); MUp = Xtmp' * X;

            L = W * S';
            for j = 1:Nmat
                L += dE * Dvec[:,j] .* MReduced * (X' * (wMat[j,:] .* Sinv .* psi))';
            end
            Wtmp, _, _ = svd([W L]); NUp = Wtmp' * W;

            X = Xtmp;
            W = Wtmp;

            S = MUp * S * (NUp')
            for j = 1:Nmat
                S += dE * (X' * (wMat[j,:] .* Sinv .* psi)) * MReduced' * (Dvec[:,j] .* W);
            end

            ############## Dose Computation ##############
            dose      .+= dEGrid * (X*S*(W'*e1) + psi*M1) * ∫Y₀⁰dΩ
            dose_coll .+= dEGrid * (X*S*(W'*e1))           * ∫Y₀⁰dΩ

            X, S, W = truncate!(obj, T.(X), T.(S), T.(W));
            r = size(S, 1)
            rVec[1,n] = energy[n];
            rVec[2,n] = r;

            next!(prog)
        end
    end

    U, Sigma, V = svd(Matrix(S))
    return Matrix(X)*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*Matrix(W)*V,
           Matrix(W)*V, Vector(dose), Vector(dose_coll), rVec, Matrix(psi)
end

function RunTracer_UniDirectional(obj::SolverCPU{T}, model::String, trace::Bool) where {T<:AbstractFloat}
    ## this function has been severely reduced since we only load precomputed tracer results in this version
    nE = 1
    E_tracer = []
    tracerDirs = [obj.settings.Omega1 obj.settings.Omega2 obj.settings.Omega3]
    nB = size(tracerDirs,1)

    if trace
        #Update scattering coefficients
        E = [0.000999999000000000,	0.00110000000000000,	0.00120000000000000, 0.00130000000000000,	0.00140000000000000,	0.00150000000000000,	0.00160000000000000,	0.00170000000000000,	0.00180000000000000,	0.00200000000000000,	0.00225000000000000,	0.00250000000000000,	0.00275000000000000,	0.00300000000000000,	0.00325000000000000,	0.00350000000000000,	0.00375000000000000,	0.00400000000000000,	0.00450000000000000,	0.00500000000000000,	0.00550000000000000,	0.00600000000000000,	0.00650000000000000,	0.00700000000000000,	0.00800000000000000,	0.00900000000000000,	0.0100000000000000,	0.0110000000000000,	0.0120000000000000,	0.0130000000000000,	0.0140000000000000,	0.0150000000000000,	0.0160000000000000,	0.0170000000000000,	0.0180000000000000,	0.0200000000000000,	0.0225000000000000,	0.0250000000000000,	0.0275000000000000,	0.0300000000000000,	0.0325000000000000,	0.0350000000000000,	0.0375000000000000,	0.0400000000000000,	0.0450000000000000,	0.0500000000000000,	0.0550000000000000,	0.0600000000000000,	0.0650000000000000,	0.0700000000000000,	0.0800000000000000,	0.0900000000000000,	0.100000000000000,	0.110000000000000,	0.120000000000000,	0.130000000000000,	0.140000000000000,	0.150000000000000,	0.160000000000000,	0.170000000000000,	0.180000000000000,	0.200000000000000,	0.225000000000000,	0.250000000000000,	0.275000000000000,	0.300000000000000,	0.325000000000000,	0.350000000000000,	0.375000000000000,	0.400000000000000,	0.450000000000000,	0.500000000000000,	0.550000000000000,	0.600000000000000,	0.650000000000000,	0.700000000000000,	0.800000000000000,	0.900000000000000,	1,	1.10000000000000,	1.20000000000000,	1.30000000000000,	1.40000000000000,	1.50000000000000,	1.60000000000000,	1.70000000000000,	1.80000000000000,	2,	2.25000000000000,	2.50000000000000,	2.75000000000000,	3,	3.25000000000000,	3.50000000000000,	3.75000000000000,	4,	4.50000000000000,	5,	5.50000000000000,	6,	6.50000000000000,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	20,	22.5000000000000,	25,	27.5000000000000,	30,	32.5000000000000,	35,	37.5000000000000,	40,	45,	50,	55,	60,	65,	70,	80,	90,	100,	110,	120,	130,	140,	150,	160,	170,	180,	200,	225,	250];
        _ = computeOutscattering(obj.csd,T.(E.+obj.settings.eRest),T.(obj.settings.OmegaMin),"gaussIntTracer")
    end

    @timeit to "Ray-tracer" begin

        if trace 
            println("Tracer not included in this version, load existing tracer result file from \"src/tracer_results\"!")
        end

        root = pkgdir(TITUS) 
        if isfile(joinpath(root, "tracer_results/$(obj.settings.tracerFileName).bin"))
            phiTracer = Array{Float64}(undef,Int.(stat(joinpath(root, "tracer_results/$(obj.settings.tracerFileName).bin")).size/8))
            io_phi = open(joinpath(root, "tracer_results/$(obj.settings.tracerFileName).bin"), "r")
        else
            error(joinpath(root, "tracer_results/$(obj.settings.tracerFileName).bin"));
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
    return E_tracer[end:-1:1], phiTracer[:,end:-1:1,1:nB]
end

function truncate!(obj::SolverCPU{T},X::Array{T,2},S::Array{T,2},W::Array{T,2}) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.r;
    rMinTotal = 2;

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
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
    return X*U[:, 1:rmax], diagm(D[1:rmax]), W*V[:, 1:rmax];
end
