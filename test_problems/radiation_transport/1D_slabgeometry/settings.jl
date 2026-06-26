__precompile__

mutable struct settings
        ## Settings of the staggered grids
    # Number of spatial vertices
    Nx::Int64;
    # Number of cell centres
    NxC::Int64;
    # Start and end point of spatial domain
    a::Float64;
    b::Float64;
    # Grid cell width
    dx::Float64;
    
    ## Settings of temporal domain
    # End time
    Tend::Float64;
    # Time increment width
    dt::Float64;
    # CFL number 
    cfl::Float64; # CFL condition

    ## Settings for angular approximation
    # Number of moments
    nPN::Int64;
    
    ## Spatial grid
    x
    xMid

    ## Problem Settings
    problem::String;   

    ## Initial conditions
    ICType::String;
    BCType::String;

    ## Physical parameters
    sigmaS::Float64; ## Try to change this to get a non-constant value for the scattering coefficients
    sigmaA::Float64;

    ## Dynamcial low-rank approximation
    r::Int; # rank of the system
    epsAdapt::Float64; # Tolerance for rank adaptive BUG integrator

    function settings(Nx::Int=1001,nPN::Int=501)
        # Setup spatial grid
        NxC = Nx + 1;
        
      
        # xMid = x .+ dx/2;
        # xMid = xMid[1:(end-1)];
        problem = "GaussianPulse"

        # Problem 
        if problem == "Planesource" #  1DPlanesource
            a = -1.5 
            b = 1.5
            # Scattering and absorption coefficients
            sigmaA = 0.0;
            sigmaS = 1.0;

            # Initial and Boundary condition
            ICType = "LS";
            BCType = "exact";

            # Defining the constants related to the simulation
        
            # Setup temporal discretisation
            Tend = 1.0;
        elseif problem == "PlanePulse"
            a = -1.0 #0.0; # Starting point for the spatial interval
            b = 1.0 #0.002; # End point for the spatial interval
            sigmaA = 1.0;
            sigmaS = 1.0;

            # Initial and Boundary condition
            ICType = "LS";
            BCType = "exact";

            # Defining the constants related to the simulation
        
            # Setup temporal discretisation
            Tend = 1.0;
        elseif problem == "GaussianPulse"
            a = -3.0 #0.0; # Starting point for the spatial interval
            b = 3.0 #0.002; # End point for the spatial interval
            sigmaA = 1.0;
            sigmaS = 1.0;

            # Initial and Boundary condition
            ICType = "LS";
            BCType = "exact";

            # Defining the constants related to the simulation
        
            # Setup temporal discretisation
            Tend = 1.0;
        end
        x = collect(range(a,stop = b,length = Nx));
        dx = x[2] - x[1];
        xMid = [x[1]-dx;x];
        xMid = xMid .+ dx/2
        
        cfl = 1.0; # CFL condition hyperbolic
        
        dt = cfl*dx;
        
        # Settings for BUG integrator
        r = 30;
        epsAdapt = 0.05; # Tolerance for rank adaptive integrator

        new(Nx,NxC,a,b,dx,Tend,dt,cfl,nPN,x,xMid,problem,ICType,BCType,sigmaS,sigmaA,r,epsAdapt);
    end 
end

function IC(obj::settings,sample::Array{Float64,1},uncertParam::String)
    x = obj.x;
    y = zeros(size(obj.x));
    if obj.problem == "Planesource"
        floor = 1e-4;
        if uncertParam == "0"
            s1 = 0.03;
            s2 = s1^2;
            for j = 1:length(y);
                y[j] = max(floor,1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-sample[1])*(x[j]-sample[1]))/2.0/s2))
            end
        elseif uncertParam == "1"
            x0 = 0.0
            s1 = 0.03;
            s2 = s1^2;
            for j = 1:length(y);
                # y[j] = 1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2);
                y[j] = max(floor,sample[1]*1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
            end
        elseif uncertParam == "2"
            x0 = sample[2]
            s1 = 0.03;
            s2 = s1^2;
            for j = 1:length(y);
                # y[j] = 1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2);
                y[j] = max(floor,sample[1]*1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
            end
        else
            println("Choose between 0 and 1 for the uncertParam as a String")
        end
    elseif obj.problem == "PlanePulse"
        if uncertParam == "0"
            for j = 1:length(y)
                if x[j] == 0
                    y[j] = 1.0
                end
            end
        end
    elseif obj.problem == "GaussianPulse"
        if uncertParam == "0"
            σ = 0.5;
            for j = 1:length(y)
                y[j] = exp(-x[j]^2/σ^2)/sqrt(2)
            end
        end
    else
        println("Initial condition not coded yet")
        
    end
    return y;
end