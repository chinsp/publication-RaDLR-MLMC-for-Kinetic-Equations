module FEM_tracer

using BandedMatrices
using DelimitedFiles
using FastGaussQuadrature

export E_dep_perGrp, project_gaussian_on_dg
export calc_M, build_G_band


# Mass matrix
function calc_M(dE::Vector{Float64}, order::Int)
    # Legendre norm coefficients 1/(2k+1) for k = 0, 1, 2
    Mcoeff = (1.0, 1/3, 1/5)

    no_grps  = length(dE)
    no_nod_E = order + 1
    N        = no_nod_E * no_grps

    M = BandedMatrix(zeros(N, N), (0, 0))

    for gr in 1:no_grps
        base = (gr - 1) * no_nod_E
        for i in 0:order
            idx = base + i + 1
            M[idx, idx] = Mcoeff[i+1] * dE[gr]
        end
    end

    return M
end
function build_G_band(E_bounds, dE, S_star, T_star, Sigma_abs, order)

    no_grps  = length(dE)
    no_nod_E = order + 1
    N        = no_grps * no_nod_E

    kl = 2 * no_nod_E - 1
    ku = 2 * no_nod_E - 1

    G = BandedMatrix(zeros(N, N), (kl, ku))

    penalty = (order + 1.0)^2 / 2.0

    A  = zeros(no_nod_E, no_nod_E)   # diagonal block   (group gr)
    Ah = zeros(no_nod_E, no_nod_E)   # high-E neighbour block (gr-1)
    Al = zeros(no_nod_E, no_nod_E)   # low-E  neighbour block (gr+1)

    for gr in 1:no_grps

        E_low  = E_bounds[gr+1]
        E_high = E_bounds[gr]

        S_low  = S_star[gr+1];  S_high = S_star[gr]
        S_A    = (S_high + S_low) * 0.5
        S_E    = (S_high - S_low) * 0.5
        # S_Q = 0 always

        T_low  = T_star[gr+1];  T_high = T_star[gr]
        T_0    = (T_high + T_low) * 0.5
        T_1    = (T_high - T_low) * 0.5
        # T_2 = 0 always

        Σ_low  = Sigma_abs[gr+1]; Σ_high = Sigma_abs[gr]
        Σ_0    = (Σ_high + Σ_low) * 0.5
        Σ_1    = (Σ_high - Σ_low) * 0.5
        # Σ_2 = 0 always

        fill!(A,  0.0)
        fill!(Ah, 0.0)
        fill!(Al, 0.0)

        # ── Removal ──────────────────────────────────────────────────────────
        # p0 equation
        A[1,1] += dE[gr] * Σ_0
        A[1,2] += dE[gr] * Σ_1 / 3.0
        # p1 equation
        A[2,1] += dE[gr] * Σ_1 / 3.0
        A[2,2] += dE[gr] * Σ_0 / 3.0

        if order == 2
            # Fortran: Σ_2 == 0, so several terms vanish.
            # Only the non-zero ones are:
            #   A_group(3,3) += dE*Σ_0/5 + dE*Σ_2*2/35  → second term zero
            A[3,3] += dE[gr] * Σ_0 / 5.0

            # These are zero because Σ_2 = 0, but kept for clarity:
            # A[1,3] += dE[gr] * Σ_2 / 5.0          # → 0
            # A[2,2] += dE[gr] * Σ_2 * 2.0/15.0     # → 0
            # A[2,3] += dE[gr] * Σ_1 * 2.0/15.0     # ← Σ_1 not Σ_2 in Fortran
            # A[3,1] += dE[gr] * Σ_2 / 5.0          # → 0
            # A[3,2] += dE[gr] * Σ_1 * 2.0/15.0
            # Fortran line: A_group(2,3) += dE*Σ_1*2/15  (Σ_a_1, not Σ_a_2)
            A[2,3] += dE[gr] * Σ_1 * 2.0/15.0
            A[3,2] += dE[gr] * Σ_1 * 2.0/15.0
        end

        # ── Continuous slowing-down (CSD) ─────────────────────────────────
        # p0 equation
        A[1,1]  += S_low
        A[1,2]  -= S_low
        Ah[1,1] -= S_high
        Ah[1,2] += S_high

        # p1 equation
        A[2,1]  += -S_low + 2.0*S_A
        A[2,2]  +=  S_low + (2.0/3.0)*S_E
        Ah[2,1] -= S_high
        Ah[2,2] += S_high

        if order == 2
            # extra terms in p0 equation
            A[1,3]  += S_low
            Ah[1,3] -= S_high

            # extra terms in p1 equation  (S_Q = 0 → 2/5*S_Q = 0)
            A[2,3]  += -S_low          # + 2/5*S_Q = 0
            Ah[2,3] -= S_high

            # p2 equation
            A[3,1]  +=  S_low + 2.0*S_E
            A[3,2]  += -S_low + 2.0*S_A   # + 4/5*S_Q = 0
            A[3,3]  +=  S_low + (4.0/5.0)*S_E
            Ah[3,1] -= S_high
            Ah[3,2] += S_high
            Ah[3,3] -= S_high
        end

        # ── Straggling – volume term ──────────────────────────────────────
        A[2,2] += 4.0 * T_0 / dE[gr]

        if order == 2
            A[2,3] += 4.0 * T_1 / dE[gr]
            A[3,2] += 4.0 * T_1 / dE[gr]
            # T_2 = 0 → 24*T_2/(5*dE) = 0
            A[3,3] += 12.0 * T_0 / dE[gr]
        end

        # ── Straggling – penalty on high-E side (g-1/2) ──────────────────
        if gr != 1
            min_dE = min(dE[gr], dE[gr-1])
            p = penalty * T_high / min_dE

            A[1,1]  += p;   A[1,2]  += p
            A[2,1]  += p;   A[2,2]  += p
            Ah[1,1] -= p;   Ah[1,2] += p
            Ah[2,1] -= p;   Ah[2,2] += p

            if order == 2
                A[1,3]  += p;   A[2,3]  += p
                A[3,1]  += p;   A[3,2]  += p;  A[3,3]  += p
                Ah[1,3] -= p;   Ah[2,3] -= p
                Ah[3,1] -= p;   Ah[3,2] += p;  Ah[3,3] -= p
            end
        end

        # ── Straggling – penalty on low-E side (g+1/2) ───────────────────
        if gr != no_grps
            min_dE = min(dE[gr], dE[gr+1])
            p = penalty * T_low / min_dE

            A[1,1]  += p;   A[1,2]  -= p
            A[2,1]  -= p;   A[2,2]  += p
            Al[1,1] -= p;   Al[1,2] -= p
            Al[2,1] += p;   Al[2,2] += p

            if order == 2
                A[1,3]  += p;   A[2,3]  -= p
                A[3,1]  += p;   A[3,2]  -= p;  A[3,3]  += p
                Al[1,3] -= p;   Al[2,3] += p
                Al[3,1] -= p;   Al[3,2] -= p;  Al[3,3] -= p
            end
        end

        # ── Straggling – consistency/symmetry on high-E side (g-1/2) ─────
        # THIS BLOCK IS MISSING FROM THE ORIGINAL JULIA PORT
        if gr != 1
            A[1,2]  -= T_high / dE[gr]
            A[2,1]  -= T_high / dE[gr]
            A[2,2]  -= T_high / dE[gr] + T_high / dE[gr]   # two separate Fortran lines

            Ah[1,2] -= T_high / dE[gr-1]
            Ah[2,1] += T_high / dE[gr]
            Ah[2,2] -= T_high / dE[gr-1] + T_high / dE[gr]

            if order == 2
                # p0 equation
                A[1,3]  -= 3.0 * T_high / dE[gr]
                Ah[1,3] += 3.0 * T_high / dE[gr-1]

                # p1 equation
                A[2,3]  -= 3.0 * T_high / dE[gr] + T_high / dE[gr]
                Ah[2,3] += 3.0 * T_high / dE[gr-1] + T_high / dE[gr]

                # p2 equation
                A[3,1]  -= 3.0 * T_high / dE[gr]
                A[3,2]  -= T_high / dE[gr] + 3.0 * T_high / dE[gr]
                A[3,3]  -= 3.0 * T_high / dE[gr] + 3.0 * T_high / dE[gr]

                Ah[3,1] += 3.0 * T_high / dE[gr]
                Ah[3,2] -= T_high / dE[gr-1] + 3.0 * T_high / dE[gr]
                Ah[3,3] += 3.0 * T_high / dE[gr-1] + 3.0 * T_high / dE[gr]
            end
        end

        # ── Straggling – consistency/symmetry on low-E side (g+1/2) ──────
        # THIS BLOCK IS MISSING FROM THE ORIGINAL JULIA PORT
        if gr != no_grps
            A[1,2]  += T_low / dE[gr]
            A[2,1]  += T_low / dE[gr]
            A[2,2]  -= T_low / dE[gr] + T_low / dE[gr]     # two separate Fortran lines

            Al[1,2] += T_low / dE[gr+1]
            Al[2,1] -= T_low / dE[gr]
            Al[2,2] -= T_low / dE[gr+1] + T_low / dE[gr]

            if order == 2
                # p0 equation
                A[1,3]  -= 3.0 * T_low / dE[gr]
                Al[1,3] += 3.0 * T_low / dE[gr+1]

                # p1 equation
                A[2,3]  += 3.0 * T_low / dE[gr] + T_low / dE[gr]
                Al[2,3] -= 3.0 * T_low / dE[gr+1] + T_low / dE[gr]

                # p2 equation
                A[3,1]  -= 3.0 * T_low / dE[gr]
                A[3,2]  += T_low / dE[gr] + 3.0 * T_low / dE[gr]
                A[3,3]  -= 3.0 * T_low / dE[gr] + 3.0 * T_low / dE[gr]

                Al[3,1] += 3.0 * T_low / dE[gr]
                Al[3,2] += T_low / dE[gr+1] + 3.0 * T_low / dE[gr]
                Al[3,3] += 3.0 * T_low / dE[gr+1] + 3.0 * T_low / dE[gr]
            end
        end

        # ── Assemble into G ───────────────────────────────────────────────
        row = (gr - 1) * no_nod_E + 1
        G[row:row+no_nod_E-1, row:row+no_nod_E-1] .+= A

        if gr > 1
            col = (gr - 2) * no_nod_E + 1
            G[row:row+no_nod_E-1, col:col+no_nod_E-1] .+= Ah
        end

        if gr < no_grps
            col = gr * no_nod_E + 1
            G[row:row+no_nod_E-1, col:col+no_nod_E-1] .+= Al
        end
    end

    return G
end
# # G matrix assembly
# function build_G_band(E_bounds, dE, S_star, T_star, Sigma_abs, order)

#     no_grps  = length(dE)
#     no_nod_E = order + 1
#     N        = no_grps * no_nod_E

#     kl = 2 * no_nod_E - 1
#     ku = 2 * no_nod_E - 1

#     G = BandedMatrix(zeros(N, N), (kl, ku))

#     penalty = (order + 1.0)^2 / 2.0
    
#     A  = zeros(no_nod_E, no_nod_E)
#     Ah = zeros(no_nod_E, no_nod_E)
#     Al = zeros(no_nod_E, no_nod_E)
#     for gr in 1:no_grps

#         E_low  = E_bounds[gr+1]
#         E_high = E_bounds[gr]

#         S_low  = S_star[gr+1];  S_high = S_star[gr]
#         S_A    = (S_high + S_low) * 0.5
#         S_E    = (S_high - S_low) * 0.5

#         T_low  = T_star[gr+1];  T_high = T_star[gr]
#         T0     = (T_high + T_low) * 0.5
#         T1     = (T_high - T_low) * 0.5

#         Σ_low  = Sigma_abs[gr+1]; Σ_high = Sigma_abs[gr]
#         Σ0     = (Σ_high + Σ_low) * 0.5
#         Σ1     = (Σ_high - Σ_low) * 0.5

#         inv_dE = 1.0 / dE[gr]

#         fill!(A, 0.0)
#         fill!(Ah, 0.0)
#         fill!(Al, 0.0)

#         # --- Removal ---
#         A[1,1] += dE[gr] * Σ0
#         A[1,2] += dE[gr] * Σ1 * (1/3)
#         A[2,1] += dE[gr] * Σ1 * (1/3)
#         A[2,2] += dE[gr] * Σ0 * (1/3)

#         if order == 2
#             A[1,3] += dE[gr] * 0.0          # Σ2 == 0 always → no-op; kept for clarity
#             A[2,2] += 0.0
#             A[2,3] += 0.0
#             A[3,1] += 0.0
#             A[3,2] += 0.0
#             A[3,3] += dE[gr] * Σ0 * 0.2
#         end

#         # --- Continuous slowing down ---
#         A[1,1] += S_low;    A[1,2] -= S_low
#         Ah[1,1] -= S_high;  Ah[1,2] += S_high

#         A[2,1]  += -S_low + 2*S_A
#         A[2,2]  +=  S_low + (2/3)*S_E
#         Ah[2,1] -= S_high
#         Ah[2,2] += S_high

#         if order == 2
#             A[1,3]  += S_low;   Ah[1,3] -= S_high
#             A[2,3]  += -S_low;  Ah[2,3] -= S_high   # S_Q == 0

#             A[3,1]  += S_low + 2*S_E
#             A[3,2]  += -S_low + 2*S_A                # S_Q == 0
#             A[3,3]  += S_low + (4/5)*S_E
#             Ah[3,1] -= S_high
#             Ah[3,2] += S_high
#             Ah[3,3] -= S_high
#         end

#         # --- Straggling volume ---
#         A[2,2] += 4 * T0 * inv_dE

#         if order == 2
#             A[2,3] += 4 * T1 * inv_dE
#             A[3,2] += 4 * T1 * inv_dE
#             A[3,3] += 12 * T0 * inv_dE   # T2 == 0 → drop the +24*T2/(5*dE) term
#         end

#         # --- Penalty (g-1/2) ---
#         if gr != 1
#             min_dE = min(dE[gr], dE[gr-1])
#             p = penalty * T_high / min_dE

#             A[1,1] += p;   A[1,2] += p
#             A[2,1] += p;   A[2,2] += p

#             Ah[1,1] -= p;  Ah[1,2] += p
#             Ah[2,1] -= p;  Ah[2,2] += p
#         end

#         # --- Penalty (g+1/2) ---
#         if gr != no_grps
#             min_dE = min(dE[gr], dE[gr+1])
#             p = penalty * T_low / min_dE

#             A[1,1] += p;   A[1,2] -= p
#             A[2,1] -= p;   A[2,2] += p

#             Al[1,1] -= p;  Al[1,2] -= p
#             Al[2,1] += p;  Al[2,2] += p
#         end

#         # --- Assemble into G ---
#         row = (gr - 1) * no_nod_E + 1
#         G[row:row+no_nod_E-1, row:row+no_nod_E-1] .= A

#         if gr > 1
#             col = (gr - 2) * no_nod_E + 1
#             G[row:row+no_nod_E-1, col:col+no_nod_E-1] .= Ah
#         end

#         if gr < no_grps
#             col = gr * no_nod_E + 1
#             G[row:row+no_nod_E-1, col:col+no_nod_E-1] .= Al
#         end
#     end

#     return G
# end

# Project a Gaussian mixture onto the DG basis
function project_gaussian_on_dg(mu::Float64, sigma::Float64, w_E::Float64,
                                 E_bounds::Vector{Float64}, dE::Vector{Float64},
                                 M::BandedMatrix,
                                 order::Int, N_E::Int)

    nqp = 7
    points, weights = gausslegendre(nqp)

    no_grps  = length(dE)
    no_nod_E = order + 1

    phi = zeros(Float64, no_grps * no_nod_E)
    fun_E = zeros(no_nod_E)   # reuse buffer across groups and quadrature points

    for gr in 1:no_grps
        E_low  = E_bounds[gr+1]
        E_high = E_bounds[gr]
        Jac    = dE[gr] * 0.5
        half_dE = dE[gr] * 0.5

        rhs = zeros(no_nod_E)

        for qp in 1:nqp
            E_qp = E_low + (points[qp] + 1) * half_dE
            calc_shape_fun_E!(fun_E, E_qp, E_low, E_high, no_nod_E)
            g = Jac * weights[qp] * gaussianMixture_E(E_qp, mu, sigma, w_E)
            @. rhs += g * fun_E
        end

        base = (gr - 1) * no_nod_E
        for node in 1:no_nod_E
            row = base + node
            phi[row] = rhs[node] / M[row, row]
        end
    end

    return phi
end

# Shape functions 
function calc_shape_fun_E!(fun_E, E, E_min, E_max, no_nod_E; scaling=false)
    dE = E_max - E_min

    if no_nod_E == 1
        fun_E[1] = scaling ? 1.0 / dE : 1.0

    elseif no_nod_E == 2
        x = (2.0 / dE) * (E - (E_min + dE * 0.5))
        fun_E[1] = scaling ? 1.0 / dE : 1.0
        fun_E[2] = scaling ? 3.0 * x / dE : x

    elseif no_nod_E == 3
        x = (2.0 / dE) * (E - (E_min + dE * 0.5))
        fun_E[1] = scaling ? 1.0 / dE : 1.0
        fun_E[2] = scaling ? 3.0 * x / dE : x
        fun_E[3] = scaling ? 5.0 * (1.5 * x^2 - 0.5) / dE : 1.5 * x^2 - 0.5

    else
        error("no_nod_E not in valid range")
    end
    return fun_E
end

# Allocating wrapper kept for backward compatibility
function calc_shape_fun_E(E, E_min, E_max, no_nod_E; scaling=false)
    fun_E = zeros(no_nod_E)
    calc_shape_fun_E!(fun_E, E, E_min, E_max, no_nod_E; scaling)
end

@inline function gaussianMixture_E(E::Float64, mu, sigma, w)

    if mu isa AbstractArray
        s = 0.0
        @inbounds for k in eachindex(mu)
            s += w[k] * exp(-0.5 * ((E - mu[k]) / sigma[k])^2) / (sqrt(2π) * sigma[k])
        end
        return s
    else
        return w * exp(-0.5 * ((E - mu) / sigma)^2) / (sqrt(2π) * sigma)
    end
end

# Energy deposition per group
function E_dep_perGrp(phi, S_star, T_star, Sigma_abs, E_bounds, dE, order)

    no_grps  = length(dE)
    no_nod_E = order + 1

    result = zeros(Float64, no_grps)

    # Boundary flux at the lowest-energy face
    E_min = E_bounds[no_grps+1]
    S_min = S_star[no_grps+1]
    base_last = no_nod_E * (no_grps - 1)

    φ0 = phi[base_last + 1]
    φ1 = phi[base_last + 2]
    E_dep = E_min * S_min * (φ0 - φ1)
    if order == 2
        φ2 = phi[base_last + 3]
        E_dep += E_min * S_min * φ2
    end
    result[1] = E_dep

    for gr in 1:no_grps
        E_low  = E_bounds[gr+1]
        E_high = E_bounds[gr]
        E_g    = (E_high + E_low) * 0.5

        S_low  = S_star[gr+1];    S_high = S_star[gr]
        S0     = (S_high + S_low) * 0.5
        S1     = (S_high - S_low) * 0.5

        T_low  = T_star[gr+1];    T_high = T_star[gr]
        T0     = (T_high + T_low) * 0.5
        T1     = (T_high - T_low) * 0.5

        Σ_low  = Sigma_abs[gr+1]; Σ_high = Sigma_abs[gr]
        Σ0     = (Σ_high + Σ_low) * 0.5
        Σ1     = (Σ_high - Σ_low) * 0.5

        base = no_nod_E * (gr - 1)
        φ0 = phi[base + 1]
        φ1 = phi[base + 2]

        dEg   = dE[gr]
        dEg2  = dEg^2

        S_part = φ0 * S0 * dEg + φ1 * S1 * dEg * (1/3)
        T_part = 2 * T0 * φ1

        if gr < no_grps
            base_next = no_nod_E * gr
            φ0n = phi[base_next + 1]
            φ1n = phi[base_next + 2]
            T_part -= T_low * (φ0n + φ1n - φ0 + φ1)
        end

        Σ_part = Σ0 * φ0 * dEg * E_g +
                 (Σ0 * φ1 + Σ1 * φ0) * dEg2 * (1/6) +
                 Σ1 * φ1 * E_g * dEg * (1/3)

        if order == 2
            φ2 = phi[base + 3]

            # S2 == 0 → S_part += 0
            T_part += 2 * T1 * φ2

            if gr < no_grps
                φ2n = phi[no_nod_E * gr + 3]
                T_part -= T_low * (φ2n - φ2)
            end
        end

        result[gr] = S_part + T_part + Σ_part
    end

    return result
end

function legendreP(n::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return x
    return ((2n - 1) * x * legendreP(n-1, x) - (n-1) * legendreP(n-2, x)) / n
end

function band_storage(G, kl, ku)
    N    = size(G, 1)
    band = zeros(2*kl + ku + 1, N)
    for col in 1:N
        for row in max(1, col-ku):min(N, col+kl)
            band[kl + ku + 1 + row - col, col] = G[row, col]
        end
    end
    return band
end

end # module