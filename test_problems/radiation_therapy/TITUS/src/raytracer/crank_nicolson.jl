module CN

using LinearAlgebra
using BandedMatrices

export construct_ODE_iter_matrices, CN_1step!

# Construct Crank–Nicolson iteration matrices
function construct_ODE_iter_matrices(G, M, dx; θ=0.5)
    inv_dx = inv(dx)

    K_iter = inv_dx .* M .+ θ       .* G
    K_rhs  = inv_dx .* M .- (1 - θ) .* G

    K_iter_LU = lu(K_iter)

    return K_iter_LU, K_rhs
end


function CN_1step!(rhs::Vector{Float64}, K_iter_LU, K_rhs, sol_old; adj_source=nothing)
     mul!(rhs, K_rhs, sol_old)  

    if adj_source !== nothing
        rhs .+= adj_source
    end

    sol = K_iter_LU \ rhs

    @. rhs = (sol_old + sol) * 0.5

    return sol, rhs   # rhs now holds sol_avg
end

end