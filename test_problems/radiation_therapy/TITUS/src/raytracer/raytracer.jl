module UniformRayTracer

export get_ray
using LinearAlgebra

function get_ray(bounds_min, bounds_max, Ncells,
                 x_start, x_end;
                 max_segments=1000)

    dir = x_end .- x_start
    ray_len = norm(dir)

    if ray_len == 0
        return false, 0, Int[], Float64[], 0.0, 0.0, zeros(3, 0)
    end

    # dir ./= ray_len

    Δ = (bounds_max .- bounds_min) ./ Ncells

    # bounding-box slab intersection 
    tmin = -Inf
    tmax =  Inf

    for d in 1:3
        if abs(dir[d]) < 1e-14
            if x_start[d] < bounds_min[d] || x_start[d] > bounds_max[d]
                return false, 0, Int[], Float64[], 0.0, 0.0, zeros(3, 0)
            end
        else
            t1 = (bounds_min[d] - x_start[d]) / dir[d]
            t2 = (bounds_max[d] - x_start[d]) / dir[d]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        end
    end

    if tmax < tmin
        return false, 0, Int[], Float64[], 0.0, 0.0, zeros(3, 0)
    end
    tmax = min(tmax, 1.0)  

    l_pre  = max(tmin, 0.0) * ray_len
    l_post = max(0.0, 1.0 - tmax) * ray_len  

    t   = max(tmin, 0.0)
    pos = x_start .+ t .* dir

    # starting voxel (clamped to valid range)
    idx = clamp.(floor.(Int, (pos .- bounds_min) ./ Δ) .+ 1, 1, Ncells)

    # per-axis step direction and DDA parameters
    step = ntuple(d -> Int(sign(dir[d])), 3)

    # Pre-compute tMax and tDelta as mutable 3-element arrays
    tMax   = MutableVector3(0.0, 0.0, 0.0)
    tDelta = MutableVector3(0.0, 0.0, 0.0)

    for d in 1:3
        if abs(dir[d]) < 1e-14
            tMax[d]   = Inf
            tDelta[d] = Inf
        else
            nb = dir[d] > 0 ? bounds_min[d] + idx[d] * Δ[d] :
                              bounds_min[d] + (idx[d] - 1) * Δ[d]
            tMax[d]   = (nb - pos[d]) / dir[d]
            tDelta[d] = Δ[d] / abs(dir[d])
        end
    end

    # Pre-allocate output buffers
    elem_ids       = Vector{Int}(undef, max_segments)
    l_list         = Vector{Float64}(undef, max_segments)
    coord_crossings = Matrix{Float64}(undef, 3, max_segments + 1)
    coord_crossings[:, 1] = pos

    seg = 0

    while seg < max_segments
        # Find the axis with the smallest tMax 
        d = tMax[1] <= tMax[2] ? (tMax[1] <= tMax[3] ? 1 : 3) :
                                 (tMax[2] <= tMax[3] ? 2 : 3)

        t_next = t + tMax[d]

        if t_next > tmax
            seg += 1
            l_list[seg] = (tmax - t) * ray_len
            elem_ids[seg]            = linear_index(idx, Ncells)
            coord_crossings[:, seg+1] = x_start .+ tmax .* dir
            break
        end

        seg += 1
        l_list[seg] = tMax[d] * ray_len
        elem_ids[seg] = linear_index(idx, Ncells)

        t += tMax[d]
        coord_crossings[:, seg+1] = x_start .+ t .* dir

        idx[d] += step[d]

        if idx[d] < 1 || idx[d] > Ncells[d]
            break
        end

        # Subtract the consumed tMax from all axes, then reset the crossed axis
        tMax[1] -= tMax[d]; tMax[2] -= tMax[d]; tMax[3] -= tMax[d]
        tMax[d]  = tDelta[d]
    end

    return true,
           seg,
           elem_ids[1:seg],
           l_list[1:seg],
           l_pre,
           l_post,
           coord_crossings[:, 1:seg+1]
end

# Inline linear index 
@inline function linear_index(idx, N)
    return idx[1] + (idx[2] - 1) * N[1] + (idx[3] - 1) * N[1] * N[2]
end

mutable struct MutableVector3
    x::Float64; y::Float64; z::Float64
end
@inline Base.getindex(v::MutableVector3, i::Int) = i == 1 ? v.x : i == 2 ? v.y : v.z
@inline function Base.setindex!(v::MutableVector3, val, i::Int)
    i == 1 ? (v.x = val) : i == 2 ? (v.y = val) : (v.z = val)
end

end