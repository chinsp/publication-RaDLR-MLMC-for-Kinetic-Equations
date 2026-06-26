__precompile__
using Interpolations
include("MaterialParametersProtons.jl")

struct CSD{T<:AbstractFloat}
    # energy grid
    eGrid::Array{T,1};
    # transformed energy grid
    eTrafo::Array{T,1};
    # tabulated energy for sigma/stopping power
    E_Tab::Array{T,1};
    E_tab_TOPAS::Array{T,1};
    # tabulated sigma
    sigma_ce::Array{T,3};
    sigma_xi::Array{T,3};
    # composition field
    comp_vector::Array{Float64,2};
    materials
    interpolants
    # settings
    settings::Settings

    # constructor
    function CSD(settings::Settings,T::DataType=Float64)
        # read tabulated material parameters
        if settings.particle =="Protons"
            param = MaterialParametersProtons(settings::Settings,settings.OmegaMin);
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            sigma_ce = param.sigma_ce;
            sigma_xi = param.sigma_xi;
            comp_vector = param.comp_vector
            E_tab_TOPAS = param.E_tab_TOPAS
            materials = param.materials
        else #Electrons (not supported at the moment!!)
            println("Only protons supported in this TITUS version!")
        end
        nTab = length(E_tab)
        E_transformed = zeros(nTab)
        for i = 2:nTab
            E_transformed[i] = E_transformed[i - 1] + ( E_tab[i] - E_tab[i - 1] ) / 2 * ( 1.0 / S_tab[i] + 1.0 / S_tab[i - 1] );
        end

        # define minimal and maximal energy for computation
        minE = settings.eMin .+ settings.eRest;
        maxE = settings.eMax;

        eTrafoMax = integrate(E_tab, 1 ./S_tab[:,1])
        eTrafo1 = zeros(nTab)

        eTrafo1[1] = eTrafoMax;
        for i = 2:length(E_tab)
            eTrafo1[i] = eTrafoMax  - integrate(E_tab[1:i], 1 ./S_tab[1:i,1])
        end
        
        ETab2ETrafo = LinearInterpolation(E_tab, eTrafo1; extrapolation_bc=Flat())
        eMaxTrafo = ETab2ETrafo( maxE );
        eMinTrafo = ETab2ETrafo( minE );
        nEnergies = Integer(ceil((maxE-minE)/settings.dE));
        if ~iseven(nEnergies)
            nEnergies = nEnergies + 1;
        end
        eGrid = collect(range(minE+settings.dE/2,maxE-settings.dE/2,length=nEnergies))[end:-1:1]
        dEGrid=zeros(length(eGrid)-1)
        for i=2:length(eGrid)
            dEGrid[i-1] = eGrid[i-1] - eGrid[i]
        end
        ETrafo2ETab = LinearInterpolation(eTrafo1[end:-1:1], E_tab[end:-1:1].-settings.eRest; extrapolation_bc=Flat())
        eTrafo = ETab2ETrafo(eGrid)

        # In your CSD struct / constructor, precompute:
        material_keys = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca"]
        interpolants = [LinearInterpolation(E_tab_TOPAS, materials[key][1];extrapolation_bc=Flat()) for key in material_keys]

        new{T}(eGrid,eTrafo,E_tab,E_tab_TOPAS,sigma_ce,sigma_xi,comp_vector,materials,interpolants,settings);
    end
end

function XiAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_xi,2)
    y = zeros(2,nPsi)
    E2Sigma_xi1 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,1],(Gridded(Linear()),NoInterp()))
    E2Sigma_xi2 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,2],(Gridded(Linear()),NoInterp()))
    y[1,:] = E2Sigma_xi1.(energy,1:nPsi)
    y[2,:] = E2Sigma_xi2.(energy,1:nPsi)
    return T.(y);
end

function SigmaAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_ce,3)
    y = zeros(obj.settings.nPN+1,nPsi)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
        E2Sigma_ce = interpolate((obj.E_Tab,1:nPsi), obj.sigma_ce[:,i,:],(Gridded(Linear()),NoInterp()))
        y[i,:] = E2Sigma_ce.(energy,1:nPsi);
    end
    return T.(y);
end

function setup_RTEnergyGrps(no_grps, E_min, E_max; spacing=:uniform)

    if spacing == :uniform
        # Uniform spacing
        E_bounds = collect(range(E_min, E_max, length=no_grps+1))

    elseif spacing == :uniformSafe #make smallest energy group smaller and add buffer to Emax to avoid issues with interpolation later
        edge_frac=0.01
        edge_frac_high=1.02
        # base uniform spacing
        Δ = (E_max - E_min) / no_grps
        δ = edge_frac * Δ

        E_bounds = zeros(no_grps+1)

        # first small bin
        E_bounds[1] = E_min
        E_bounds[2] = E_min + δ

        # interior bins (uniform)
        for i in 3:no_grps
            E_bounds[i] = E_bounds[i-1] + Δ
        end
        E_bounds[end] = E_max*edge_frac_high
    elseif spacing == :log
        # Logarithmic spacing
        E_bounds = exp.(range(log(E_min), log(E_max), length=no_grps+1))

    else
        error("spacing must be :uniform, :uniformSafe or :log")
    end

    # Group widths
    dE = diff(E_bounds)

    # Midpoints
    E_mid = (E_bounds[1:end-1] .+ E_bounds[2:end]) ./ 2

    return E_bounds[end:-1:1], E_mid[end:-1:1], dE[end:-1:1]
end

function computeOutscattering(obj::CSD{T}, energy::Array{T,1},minOmega,type::String)
    param = MaterialParametersProtons(obj.settings,minOmega);
    E_tab = param.E_tab;
    sigma_ce = param.sigma_ce;
    xi=zeros(length(energy),12)
    root = pkgdir(TITUS)

    if type == "gaussIntTracer"
        nE = size(energy,1)        
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        if minOmega > 0
            w[mu.>cosd(minOmega)] .= 0;
        end
        nMat = 12
        N = size(obj.sigma_ce,2) 
        open(joinpath(root, "src/raytracer/proton_totalXS_data"), "w") do file
            println(file, nE)
            for i=1:nE
                println(file, energy[i]-938.26)
            end
            for k=1:nMat
                println(file,matNames[k])
                for n = 1:nE
                    xi_e = 2*pi*dot(w,sigma_ce[n,:,k])
                    println(file,xi_e) 
                end
            end
        end
    elseif type == "gaussInt" #doesnt write files for tracer
        nE = size(energy,1)
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        nMat = 12
        for k=1:nMat
            for n = 1:nE
                xi[n,k] = 2*pi*dot(w,sigma_ce[n,:,k])
            end
        end
    elseif type == "FP_correction"
        N=19
        nE = size(energy,1)
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        nMat = 12
        N = size(obj.sigma_ce,2)
        open(joinpath(root, "src/raytracer/proton_totalXS_data"), "w") do file
            println(file, nE)
            for i=1:nE
                println(file, energy[i]-938.26)
            end
            for k=1:nMat
                println(file,matNames[k])
                for n = 1:nE
                    xi_e = 2*pi*dot(w,sigma_ce[n,:,k].*(1 .- mu)) 
                    println(file,0.5*xi_e*(N*(N+1))) 
                end
            end
        end
    end
    return xi*param.comp_vector
end

function get_S_at_energy(obj::CSD{T}, E::T, density) where {T<:AbstractFloat}
    S_at_E = [itp(E) for itp in obj.interpolants]

    idx = Base.unique(i -> density[i], 1:length(density))
    stp = zeros(length(idx))
    for (k, j) in enumerate(idx)
        stp[k] = (obj.comp_vector[:, k]' * S_at_E) * density[j] / 100.0
    end
    return stp
end