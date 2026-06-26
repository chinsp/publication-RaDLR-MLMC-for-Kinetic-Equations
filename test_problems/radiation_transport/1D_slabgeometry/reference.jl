using LinearAlgebra
using FastGaussQuadrature, LegendrePolynomials
using SpecialFunctions


function Θ(x) 
    # The Heaviside function 
    # Θ(x) = { 1 if x >= 0
    #          0, x < 0
    if x >= 0
        return 1.0;
    else
        return 0.0;
    end
end

## Semi-Analytic solution for the Plane Pulse Problem 
function computeSemiAnalytic_PlanePulse_uncollided_xt(t::Float64,x::Float64,c::Float64)
    η = x/t;
    q = (1+η)/(1-η);

    ϕ_u_xt = 1/2 * exp(-t)/t * Θ(1 - abs(η));
    return ϕ_u_xt;
end

function computeSemiAnalytic_PlanePulse_uncollided(t::Float64,x::Array{Float64,1},c::Float64)
    ϕ_u = zeros(length(x));
    for i = eachindex(x)
        ϕ_u[i] = computeSemiAnalytic_PlanePulse_uncollided_xt(t,x[i],c);
    end
    return ϕ_u;
end

function computeSemiAnalytic_PlanePulse_collided_xt(t::Float64,x::Float64,c::Float64)
    η = x/t;
    q = (1+η)/(1-η);

    n = 1001;
    μ,w = gausslegendre(n)

    u = pi/2 .*(μ .+ 1);

    ξ(v) = (log(q) + v*im)/(η + im*tan(v/2));

    integral = 0.0;
    for i = eachindex(u)
        integral += w[i]*pi/2 *sec(u[i]/2)^2 * real( ξ(u[i])^2 * exp( c*t/2 * (1-η^2)*ξ(u[i]) ) );
    end

    ϕ_c_xt = c*(exp(-t)/8/pi*(1-η^2)*integral)*Θ(1 - abs(η))

    return ϕ_c_xt;
end

function computeSemiAnalytic_PlanePulse_collided(t::Float64,x::Array{Float64,1},c::Float64)
    ϕ_c = zeros(length(x));
    for i = eachindex(x)
        ϕ_c[i] = computeSemiAnalytic_PlanePulse_collided_xt(t,x[i],c);
    end
    return ϕ_c;
end

function computeSemiAnalytic_PlanePulse(t::Float64,x::Array{Float64,1},c::Float64)
    ϕ_u = computeSemiAnalytic_PlanePulse_uncollided(t,x,c);
    ϕ_c = computeSemiAnalytic_PlanePulse_collided(t,x,c);
    return ϕ_u .+ ϕ_c;
end

function computegPCCoeffs_PlanePulse_Uniform(t::Float64,x::Array{Float64,1},c::Float64,N::Int)
    ## 
    n = 101;
    θ,w = gausslegendre(n)

    a = zeros(N+1,length(x));

    for i = 0:N
        a_i = a[i+1,:]
        for k = 1:n
            ϕ_c_θ = computeSemiAnalytic_PlanePulse_collided(t,x,c*(1+θ[k]/10))
            a_i .+= w[k] .* ϕ_c_θ .*Pl(θ[k],i);
        end
        a[i+1,:] .= (2i+1)/2 .* a_i;
        println("$i th coefficient computed")
    end
    return a;
end

function computeMoments_PlanePulse_Uniform(t::Float64,x::Array{Float64,1},c::Float64,N::Int)
    ϕ_u = computeSemiAnalytic_PlanePulse_uncollided(t,x,c);

    a = computegPCCoeffs_PlanePulse_Uniform(t,x,c,N);

    mean = ϕ_u .+ a[1,:];
    var = zeros(length(x));
    for i = 1:N
        var .+= a[i+1,:].^2 ./(2i+1);
    end
    return mean, var;
end


###################################################
## Semi-Analytic solution Gaussian Pulse Problem ##
###################################################


function computeSemiAnalytic_GaussianPulse_uncollided_xt(t::Float64,x::Float64,c::Float64,σ::Float64)
    ϕ_u_xt = σ * sqrt(pi) * exp(-t) * (erf((t-x)/σ) + erf((t+x)/σ))/4/t;
    return ϕ_u_xt;
end

function computeSemiAnalytic_GaussianPulse_uncollided(t::Float64,x::Array{Float64,1},c::Float64,σ::Float64)
    ϕ_u = zeros(length(x));
    for i = eachindex(x)
        ϕ_u[i] = computeSemiAnalytic_GaussianPulse_uncollided_xt(t,x[i],c,σ);
    end
    return ϕ_u;
end

function computeSemiAnalytic_GaussianPulse_collided_xt(t::Float64,x::Float64,c::Float64,σ::Float64)
    # η = x/t;
    q(η) = (1+η)/(1-η);

    n = 101;
    μ,w = gausslegendre(n)

    u = pi/2 .*(μ .+ 1);
    s = x .+ t.*μ;

    ξ(v,η) = (log(q(η)) + v*im)/(η + im*tan(v/2));
    integral = 0.0;
    for i = eachindex(u)
        for j = eachindex(s)
            η1 = (x-s[j])/t;
            integral += w[i]*w[j]*pi/2 * t * exp(-s[j]^2/σ^2) * (1-η1^2) *(1/cos(u[i]/2))^2 * real( ξ(u[i],η1)^2 * exp( c*t * (1-η1^2)*ξ(u[i],η1)/2 ) );
        end
    end

    ϕ_c_xt = c*exp(-t)/8/pi*integral;

    return ϕ_c_xt;
end

function computeSemiAnalytic_GaussianPulse_collided(t::Float64,x::Array{Float64,1},c::Float64,σ::Float64)
    ϕ_c = zeros(length(x));
    for i = eachindex(x)                
        ϕ_c[i] = computeSemiAnalytic_GaussianPulse_collided_xt(t,x[i],c,σ);
    end
    return ϕ_c;
end

function computeSemiAnalytic_GaussianPulse(t::Float64,x::Array{Float64,1},c::Float64,σ::Float64)
    ϕ_u = computeSemiAnalytic_GaussianPulse_uncollided(t,x,c,σ);
    ϕ_c = computeSemiAnalytic_GaussianPulse_collided(t,x,c,σ);
    return ϕ_u .+ ϕ_c;
end

function computegPCCoeffs_GaussianPulse_Uniform(t::Float64,x::Array{Float64,1},c::Float64,σ::Float64,N::Int)
    ## 
    n = 101;
    θ,w = gausslegendre(n)

    a = zeros(N+1,length(x));

    ϕ_c_θ_list = zeros(n,length(x));

    for k = 1:n
        c1 = c*(1 + θ[k]/10);
        ϕ_c_θ = computeSemiAnalytic_GaussianPulse_collided(t,x,c1,σ)
        ϕ_c_θ_list[k,:] .= ϕ_c_θ;
    end

    for i = 0:N
        a_i = a[i+1,:]
        for k = 1:n
            a_i .+= w[k] .* ϕ_c_θ_list[k,:] .*Pl(θ[k],i);
        end
        a[i+1,:] .= (2*i+1)/2 .* a_i;
        println("$i th coefficient computed")
    end
    return a;
end

function computeMoments_GaussianPulse_Uniform(t::Float64,x::Array{Float64,1},c::Float64,σ::Float64,N::Int)
    ϕ_u = computeSemiAnalytic_GaussianPulse_uncollided(t,x,c,σ);

    a = computegPCCoeffs_GaussianPulse_Uniform(t,x,c,σ,N);

    mean = ϕ_u .+ a[1,:];
    var = zeros(length(x));
    for i = 1:N
        var .+= a[i+1,:].^2 ./(2i+1);
    end
    return mean, var;
end