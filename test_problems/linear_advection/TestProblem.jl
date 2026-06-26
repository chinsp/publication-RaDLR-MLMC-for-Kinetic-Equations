using PyPlot
mutable struct LinearAdvection1d
    Nx::Int;
    x0::Float64;
    xN::Float64;
    dx::Float64;
    dt::Float64;
    T::Float64;
    c::Float64;
    CFL::Float64;
    x::Array{Float64,1};
    upwindMatrixAssembly::Function;
    upwindSolve::Function;
    problem::String;
    
    function LinearAdvection1d(Nx::Int = 1000,x0::Float64 = 0.0,xN::Float64 =10.0,c::Float64=1.0,T::Float64=1.0,CFL::Float64=1.0)
        #  = (xN-x0)/Nx;
        Nx = Nx + 1;
        x=collect(range(x0,xN,Nx));
        dx = x[2] - x[1];
        dt = CFL*dx/c;
        
        upwindMatrixAssembly = function() 
            gamma_min = min(c,0.0);
            gamma_max = max(c,0.0);

            alpha_min= gamma_min*dt/dx;
            alpha_max= gamma_max*dt/dx;

            A = zeros(Nx, Nx)
            for i in 1:Nx
                A[i, i] = 1 + alpha_min - alpha_max
                if i > 1
                    A[i, i-1] = alpha_max
                end
                if i < Nx
                    A[i, i+1] = -alpha_min
                end
            end
            A[1, end] = alpha_max
            A[end, 1] = -alpha_min
            return A
        end
        
        upwindSolve = function(u0::Array{Float64,1})
             return upwindMatrixAssembly()*u0
        end        

        problem = "LinearAdvection1d"; # Problem type
        new(Nx,x0,xN,dx,dt,T,c,CFL,x,upwindMatrixAssembly,upwindSolve,problem);
    end 
end




function SetupIC(LA1D::LinearAdvection1d,alpha::Array{Float64,1})
    s1 = 0.25;
    s2 = s1^2;
    x = LA1D.x;
    x0 = -1.0;
    u0 = [max(1e-4,alpha[2]/(sqrt(2*pi)*s1) *exp(-(n-x0+alpha[1])^2/2.0/s2)) for n in x];
    uT = [max(1e-4,alpha[2]/(sqrt(2*pi)*s1) *exp(-((n-LA1D.c*LA1D.T)-x0+alpha[1])^2/2.0/s2)) for n in x];
    return u0,uT;
end

function solveLinearAdvection1d(LA1D::LinearAdvection1d,alpha::Array{Float64,1})
    # Setup initial condition
    u0,uT=SetupIC(LA1D,alpha);
    u = u0;
    # Plot initial condition
    # close("all");
    # plt.figure(figsize=(12,8));
    # x = LA1D.x;
    # plt.plot(x,u0,label="Initial value");
    # plt.plot(x,uT,label=string("Analytical solution at t=",LA1D.T));

    # Solve upwind scheme
    Nt = floor(LA1D.T/LA1D.dt);
    # println("Nt = ",Nt);
    # println("Delta T = ", LA1D.dt);
    # time = 0.0;
    for t = 1:Nt
        u .= LA1D.upwindSolve(u0);
        u0 .= u;
        # time += LA1D.dt;
    end
    # println("Time = ",time);
    # plt.plot(x,u0,label=string("Solution at t=",LA1D.T));
    # plt.legend(fontsize=20);
    # plt.grid(linestyle="dotted");
    # plt.title("Upwind scheme");
    # plt.savefig("upwind.png");
    # println(norm(uT - u0)/norm(u0));
    return u0;
end

# LA1D1=LinearAdvection1d(2^7,-2.0,2.0,1.0,1.0,1.0);
# u0 = solveLinearAdvection1d(LA1D1,[-0.10,1.2]);
# println("fin")
