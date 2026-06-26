module ProtonData

using LinearAlgebra
using Interpolations

export ProtonTables, XSDatabase, load_proton_tables,
       build_xs_database, get_material_xs

const ELEMENT_ORDER = [:H,:C,:N,:O,:Na,:Mg,:P,:S,:Cl,:Ar,:K,:Ca]

struct ProtonTables
    E_sigma::Vector{Float64}
    sigma_elements::Dict{Symbol,Vector{Float64}}

    E_S::Vector{Float64}
    S_elements::Dict{Symbol,Vector{Float64}}
end

struct XSDatabase
    HU_range::Vector{Int}        # HU values
    rho::Vector{Float64}         # densities
    comp::Matrix{Float64}        # 12 × length(HU_range)
    tables::ProtonTables
end

#atomic numbers
Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
#atomic weights
Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078] #g/Mol;
#Ionization energies
I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
I_MeV = 1.0e-6 .* I_eV
ln_I_MeV = log.(I_MeV)
N_A = 6.02214076e23 #avogadro constant
m_p = 938.272
m_e = 0.511
eps0 = 1.418284572502546E-26
ee = 1.6021766208E-19

function load_sigma_tables(file_sigma)

    open(file_sigma) do io
        n = parse(Int, readline(io))

        E = [parse(Float64, readline(io)) for _ in 1:n]

        elements = [:H,:C,:N,:O,:Na,:Mg,:P,:S,:Cl,:Ar,:K,:Ca]

        sigma = Dict{Symbol,Vector{Float64}}()

        for el in elements
            readline(io)
            sigma[el] = [parse(Float64, readline(io)) for _ in 1:n]
        end

        tables_sigma = (E, sigma)

        return tables_sigma
    end
end

function load_S_tables(file_S)

    open(file_S) do io
        n = parse(Int, readline(io))

        E = [parse(Float64, readline(io)) for _ in 1:n]

        elements = [:H,:C,:N,:O,:Na,:Mg,:P,:S,:Cl,:Ar,:K,:Ca]

        S = Dict{Symbol,Vector{Float64}}()

        for el in elements
            readline(io)
            S[el] = [parse(Float64, readline(io)) for _ in 1:n]
        end

        return (E,S)
    end
end

function load_proton_tables(file_sigma,file_S)

    Eσ, σ = load_sigma_tables(file_sigma)
    ES, S = load_S_tables(file_S)

    return ProtonTables(Eσ,σ,ES,S)

end

function sigma_total(E, rho, comp_vector, tables::ProtonTables)

    σ_sum = 0.0

    for (i, el) in enumerate(ELEMENT_ORDER)

        σ_table = tables.sigma_elements[el]
        itp = LinearInterpolation(tables.E_sigma, σ_table)

        σ_sum += comp_vector[i] * itp(E)

    end

    return rho * σ_sum / 100
end

function S_star(E, rho, comp_vector, tables::ProtonTables)

    S_sum = 0.0

    for (i, el) in enumerate(ELEMENT_ORDER)
        S_table = tables.S_elements[el]
        itp = LinearInterpolation(tables.E_S, S_table)

        S_sum += comp_vector[i] * itp(E)
    end
    S_sum = rho * S_sum / 100 
    S_star = S_sum + (1.0 / 2.0) * p_dT_dE(E,rho,comp_vector)
    return S_star
end

function p_dT_dE(E_MeV::Float64, rho::Float64, comp_vector::Vector{Float64})

    corr = 1.0 / (4.0 * pi* eps0)^2

    # number density of each element
    N_i = rho .* (comp_vector ./ 100.0) .* N_A ./ Mol_weights

    v_p = sqrt(2.0 * E_MeV / m_p)

    val = corr .* N_i .* 4.0 .* pi .* ee^4 .* Z_array .* 2.0 .* I_MeV .* m_p ./
          (3.0 .* m_e .* E_MeV^2) .* (-log.(4.0 .* m_e .* E_MeV ./ (I_MeV .* m_p)) .+ 1.0)

    return sum(val)

end

function T_star(E, rho, comp_vector)

    v_p = sqrt(2.0 * E / m_p)
    T = 0.0
    corr = 1.0/ (4.0 * pi * eps0)^2
    for i in eachindex(comp_vector)
        Z = Z_array[i]
        N = rho * comp_vector[i] / 100 * N_A / Mol_weights[i]
        
        T += corr * N* 4.0 * pi * ee^4 * Z * (1.0 + 4.0 * I_MeV[i] / (3.0 * m_e * v_p^2) * log(2.0 * m_e * v_p^2 / I_MeV[i]))
        # T += corr * N * Z * (1 + 4I_MeV[i]/(3*m_e*v^2) * log(2m_e*v^2/I_MeV[i]))
    end

    return T/2
end

function build_xs_database(tables::ProtonTables, HU_range::Vector{Int})

    rho = HUtoDensity(HU_range)    # Vector{Float64}, length = length(HU_range)
    comp = matComp(HU_range)       # Matrix{Float64}, size = 12 × length(HU_range)
    
    return XSDatabase(HU_range, rho, comp, tables)
end

function get_material_xs(db::XSDatabase, HUs::Vector{Int}, Es::Vector{Float64})
    n_HU = length(HUs)
    n_E  = length(Es)

    # Output arrays
    S_out = zeros(n_HU, n_E)
    T_out = zeros(n_HU, n_E)
    Σ_out = zeros(n_HU, n_E)

    # Loop over all HU and Energy combinations
    for i in 1:n_HU
        # Find the index in the database
        idx = findfirst(==(HUs[i]), db.HU_range)
        if idx === nothing
            error("HU value $(HUs[i]) not found in database.")
        end

        rho_val = db.rho[idx]
        comp_vec = db.comp[:, idx]

        for j in 1:n_E
            E = Es[j]
            Σ_out[i, j] = sigma_total(E, rho_val, comp_vec, db.tables)
            S_out[i, j] = S_star(E, rho_val, comp_vec, db.tables)
            T_out[i, j] = T_star(E, rho_val, comp_vec)
        end
    end

    return S_out, T_out, Σ_out
end

function matComp(HU::Array{Int,1}) 
    #this is equivalent to the mat_comp.f90 class in tracer (adapted to requirements of DLRA code)
    #computes density and material composition for given HU CT values
    HU_min = -1000
    HU_max = +1600

    #material info
    no_comp_nuclides = 12
    H_mat  =  1
    C_mat  =  2
    N_mat  =  3
    O_mat  =  4
    Na_mat =  5
    Mg_mat =  6
    P_mat  =  7
    S_mat  =  8
    Cl_mat =  9
    Ar_mat = 10
    K_mat  = 11
    Ca_mat = 12

    rho_H  = 8.3748E-05
    rho_C  = 2.0
    rho_N  = 0.0011652
    rho_O  = 0.00133151
    ho_Na = 0.971
    rho_Mg = 1.74
    rho_P  = 2.2
    rho_S  = 2.0
    rho_Cl = 0.00299473
    rho_Ar = 0.00166201
    rho_K  = 0.862
    rho_Ca = 1.55

    #Atomic numbers
    Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
    #Atomic weights
    Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078]
    #Ionization energies
    I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
    I_MeV = 1.0E-6 .* I_eV
    #Ln of ionization energies
    ln_I_MeV = log.(I_MeV)

    rho_air = 1.21E-3
    rho_adipose = 0.93

    #compute comp. vector
    comp_vector = zeros(no_comp_nuclides,length(HU))

    for  i=1:length(HU)
        if     ( (HU[i] >= -1000)&&(HU[i] <=  -950) ) 
            comp_vector[H_mat ,i] =  0.0
            comp_vector[C_mat ,i] =  0.0
            comp_vector[N_mat ,i] = 75.5
            comp_vector[O_mat ,i] = 23.3
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.0
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  1.3
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   -950)&&(HU[i] <=  -120) ) 
            comp_vector[H_mat ,i] = 10.3
            comp_vector[C_mat ,i] = 10.5
            comp_vector[N_mat ,i] =  3.1
            comp_vector[O_mat ,i] = 74.9
            comp_vector[Na_mat ,i] =  0.2
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.2
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.3
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.2
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   -120)&&(HU[i] <=   -83) ) 
            comp_vector[H_mat ,i] = 11.6
            comp_vector[C_mat ,i] = 68.1
            comp_vector[N_mat ,i] =  0.2
            comp_vector[O_mat ,i] = 19.8
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -83)&&(HU[i] <=   -53) ) 
            comp_vector[H_mat ,i] = 11.3
            comp_vector[C_mat ,i] = 56.7
            comp_vector[N_mat ,i] =  0.9
            comp_vector[O_mat ,i] = 30.8
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -53)&&(HU[i] <=   -23) ) 
            comp_vector[H_mat ,i] = 11.0
            comp_vector[C_mat ,i] = 45.8
            comp_vector[N_mat ,i] =  1.5
            comp_vector[O_mat ,i] = 41.1
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -23)&&(HU[i] <=    +7) ) 
            comp_vector[H_mat ,i] = 10.8
            comp_vector[C_mat ,i] = 35.6
            comp_vector[N_mat ,i] =  2.2
            comp_vector[O_mat ,i] = 50.9
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >     +7)&&(HU[i] <=   +18) ) 
            comp_vector[H_mat ,i] = 10.6
            comp_vector[C_mat ,i] = 28.4
            comp_vector[N_mat ,i] =  2.6
            comp_vector[O_mat ,i] = 57.8
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    +18)&&(HU[i] <=   +80) ) 
            comp_vector[H_mat ,i] = 10.3
            comp_vector[C_mat ,i] = 13.4
            comp_vector[N_mat ,i] =  3.0
            comp_vector[O_mat ,i] = 72.3
            comp_vector[Na_mat ,i] =  0.2
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.2
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.2
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    +80)&&(HU[i]<=  +120) ) 
            comp_vector[H_mat ,i] =  9.4
            comp_vector[C_mat ,i] = 20.7
            comp_vector[N_mat ,i] =  6.2
            comp_vector[O_mat ,i] = 62.2
            comp_vector[Na_mat ,i] =  0.6
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.6
            comp_vector[Cl_mat ,i] =  0.3
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   +120)&&(HU[i] <=  +200) ) 
            comp_vector[H_mat ,i] =  9.5
            comp_vector[C_mat ,i] = 45.5
            comp_vector[N_mat ,i] =  2.5
            comp_vector[O_mat ,i] = 35.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  2.1
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  4.5
        elseif ( (HU[i]>   +200)&&(HU[i] <=  +300) ) 
            comp_vector[H_mat ,i] =  8.9
            comp_vector[C_mat ,i] = 42.3
            comp_vector[N_mat ,i] =  2.7
            comp_vector[O_mat ,i] = 36.3
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  3.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  6.4
        elseif ( (HU[i] >   +300)&&(HU[i] <=  +400) ) 
            comp_vector[H_mat ,i] =  8.2
            comp_vector[C_mat ,i] = 39.1
            comp_vector[N_mat ,i] =  2.9
            comp_vector[O_mat ,i] = 37.2
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  3.9
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  8.3
        elseif ( (HU[i] >   +400)&&(HU[i] <=  +500) ) 
            comp_vector[H_mat ,i] =  7.6
            comp_vector[C_mat ,i] = 36.1
            comp_vector[N_mat ,i] =  3.0
            comp_vector[O_mat ,i] = 38.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  4.7
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 10.1
        elseif ( (HU[i] >   +500)&&(HU[i] <=  +600) ) 
            comp_vector[H_mat ,i] =  7.1
            comp_vector[C_mat ,i] = 33.5
            comp_vector[N_mat ,i] =  3.2
            comp_vector[O_mat ,i] = 38.7
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  5.4
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 11.7
        elseif ( (HU[i]>   +600)&&(HU[i] <=  +700) ) 
            comp_vector[H_mat ,i] =  6.6
            comp_vector[C_mat ,i] = 31.0
            comp_vector[N_mat ,i] =  3.3
            comp_vector[O_mat ,i] = 39.4
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  6.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 13.2
        elseif ( (HU[i] >   +700)&&(HU[i] <=  +800) ) 
            comp_vector[H_mat ,i] =  6.1
            comp_vector[C_mat ,i] = 28.7
            comp_vector[N_mat ,i] =  3.5
            comp_vector[O_mat ,i] = 40.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  6.7
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 14.6
        elseif ( (HU[i]>   +800)&&(HU[i]<=  +900) ) 
            comp_vector[H_mat ,i] =  5.6
            comp_vector[C_mat ,i] = 26.5
            comp_vector[N_mat ,i] =  3.6
            comp_vector[O_mat ,i] = 40.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  7.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 15.9
        elseif ( (HU[i] >   +900)&&(HU[i] <= +1000) ) 
            comp_vector[H_mat ,i] =  5.2
            comp_vector[C_mat ,i] = 24.6
            comp_vector[N_mat ,i] =  3.7
            comp_vector[O_mat ,i] = 41.1
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  7.8
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 17.0
        elseif ( (HU[i]>  +1000)&&(HU[i]<= +1100) ) 
            comp_vector[H_mat ,i] =  4.9
            comp_vector[C_mat ,i] = 22.7
            comp_vector[N_mat ,i] =  3.8
            comp_vector[O_mat ,i] = 41.6
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  8.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 18.1
        elseif ( (HU[i]>  +1100)&&(HU[i]<= +1200) ) 
            comp_vector[H_mat ,i] =  4.5
            comp_vector[C_mat ,i] = 21.0
            comp_vector[N_mat ,i] =  3.9
            comp_vector[O_mat ,i] = 42.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  8.8
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 19.2
        elseif ( (HU[i]>  +1200)&&(HU[i]<= +1300) ) 
            comp_vector[H_mat ,i] =  4.2
            comp_vector[C_mat ,i] = 19.4
            comp_vector[N_mat ,i] =  4.0
            comp_vector[O_mat ,i] = 42.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  9.2
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 20.1
        elseif ( (HU[i] >  +1300)&&(HU[i] <= +1400) ) 
            comp_vector[H_mat ,i] =  3.9
            comp_vector[C_mat ,i] = 17.9
            comp_vector[N_mat ,i] =  4.1
            comp_vector[O_mat ,i] = 42.9
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  9.6
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 21.0
        elseif ( (HU[i]>  +1400)&&(HU[i] <= +1500) ) 
            comp_vector[H_mat ,i] =  3.6
            comp_vector[C_mat ,i] = 16.5
            comp_vector[N_mat ,i] =  4.2
            comp_vector[O_mat ,i] = 43.2
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] = 10.0
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 21.9
        elseif ( (HU[i] >  +1500)&&(HU[i] <= +1600) ) 
            comp_vector[H_mat ,i] =  3.4
            comp_vector[C_mat ,i] = 15.5
            comp_vector[N_mat ,i] =  4.2
            comp_vector[O_mat ,i] = 43.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] = 10.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 22.5
        else
            println("CT_value out of range")
        end
    end
    return comp_vector
end

function HUtoDensity(HU::Array{Int,1}) 
        #This routine is based on that of the raytracer, which uses paper:
        # Schneider, Bortfeld and Schlegel
        # Correlation between CT numbers and tissue parameters needed for Monte Carlo
        # simulations of clinical dose distributions.
        # Phys. Med. Biol. 45 (2000), pp. 459-478.
        rho_air = 1.21E-3
        rho_adipose = 0.93

        rho_values = zeros(length(HU))
        for  i=1:length(HU)
            if     ( (HU[i] >= -1000)&&(HU[i] <=   -98) ) 
                rho_values[i] = ((HU[i] + 98.0)/(98.0 - 1000.0)) * rho_air + ((HU[i] + 1000.0)/(1000.0 - 98.0)) * rho_adipose
            elseif ( (HU[i] >    -98)&&(HU[i] <=   +14) ) 
                rho_values[i] = 1.0180 + 0.893e-3 * HU[i]
            elseif ( (HU[i] >    +14)&&(HU[i] <=   +23) ) 
                rho_values[i] = 1.030
            elseif ( (HU[i] >    +23)&&(HU[i] <=  +100) ) 
                rho_values[i] = 1.0030 + 1.169e-3 * HU[i]
            elseif ( (HU[i] >   +100)&&(HU[i] <= +1600) ) 
                rho_values[i] = 1.0170 + 0.592e-3 * HU[i]
            else
             println("Warning: CT_value out of range! Using upper/lower bound values.")
                #set value to upper/lower limit instead of 0
                if HU[i] < -1000
                    rho_values[i] = 1.21E-3 #set to air
                else 
                    rho_values[i] = 1.0170 + 0.592e-3 * HU[i] #extrapolate
                end
            end
        end
        return rho_values
    end

end #module