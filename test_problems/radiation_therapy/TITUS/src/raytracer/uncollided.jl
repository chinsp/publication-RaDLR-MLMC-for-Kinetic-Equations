module Uncollided
 
using LinearAlgebra
using FastGaussQuadrature
include("fem.jl")
using .FEM_tracer
include("crank_nicolson.jl")
using .CN
include("raytracer.jl")
using .UniformRayTracer

const order_tracing              = 2
const no_nod_E_tracing           = order_tracing + 1
const segment_length_threshold   = 1e-4
const min_step_length_tracing    = 1e-2
 
export calc_uncollided_flux, no_sub_steps_tracing
 
@inline no_sub_steps_tracing(dz::Float64) =
    dz <= min_step_length_tracing ? 1 : Int(floor(dz / min_step_length_tracing)) + 1
 
function calc_uncollided_flux(
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    bounds_min::Vector{Float64},
    bounds_max::Vector{Float64},
    Ncells::Vector{Int},
    Sigma_abs::Matrix{Float64},
    S_star::Matrix{Float64},
    T_star::Matrix{Float64},
    CTscan_values::Vector{Int},
    mean_gauss_E::Float64,
    sigma_gauss_E::Float64,
    E_bounds::Vector{Float64},
    w_E::Float64,
    N_E::Int,
    mode::Symbol = :uniform, #:uniform or :octree,
    cell_coords::Matrix{Float64} = zeros(0,0)
)
    no_dof = N_E * (order_tracing + 1)
 
    dE     = abs.(diff(E_bounds))
    M_band = calc_M(dE, order_tracing)
 
    phi_0 = project_gaussian_on_dg(mean_gauss_E, sigma_gauss_E, w_E,
                                   E_bounds, dE, M_band, order_tracing, N_E)
    
    ray_intersects, no_segments, elem_ids, l_list, l_pre, l_post, coord_crossings =
    get_ray(bounds_min, bounds_max, Ncells, x_start, x_end)

    # Pre-allocate for at most no_segments touched cells (upper bound).
    # We only fill entries for segments that pass the length threshold.
    touched_ids  = Vector{Int}(undef, no_segments)
    touched_vals = zeros(Float64, no_segments, N_E)
    n_touched    = 0
 
    phi_old     = phi_0
    last_HU_seg = -1
    K_iter_LU_c = nothing
    K_rhs_c     = nothing
    rhs_buf     = Vector{Float64}(undef, no_dof)
 
    for seg in 1:no_segments
        l = l_list[seg]
        l <= segment_length_threshold && continue
 
        elem_no  = elem_ids[seg]
        HU_seg   = CTscan_values[elem_no]
        no_steps = no_sub_steps_tracing(l)
        dz       = l / no_steps
 
        if HU_seg != last_HU_seg || no_steps == 1
            G_band = build_G_band(E_bounds, dE,
                                  S_star[HU_seg, :], T_star[HU_seg, :],
                                  Sigma_abs[HU_seg, :], order_tracing)
            K_iter_LU_c, K_rhs_c = construct_ODE_iter_matrices(G_band, M_band, dz)
            last_HU_seg = HU_seg
        end
 
        phi_avg = similar(phi_old)
        for _ in 1:no_steps
            phi_new, phi_avg = CN_1step!(rhs_buf, K_iter_LU_c, K_rhs_c, phi_old)
            phi_old = phi_new
        end
 
        n_touched += 1
        touched_ids[n_touched]     = elem_no
        touched_vals[n_touched, :] = E_dep_perGrp(phi_avg,
                                                   S_star[HU_seg, :], T_star[HU_seg, :],
                                                   Sigma_abs[HU_seg, :],
                                                   E_bounds, dE, order_tracing)
    end
 
    # Return only the filled portion — no copy of the full CT volume.
    return view(touched_ids,  1:n_touched),
           view(max.(touched_vals,0.0), 1:n_touched, :)
end
 
end # module