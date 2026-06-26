using SparseArrays

__precompile__

mutable struct settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    # grid cell width
    dx::Float64
    dy::Float64

    # time settings
    # end time
    Tend::Float64;
    # time increment
    dt::Float64;
    # CFL number 
    cfl::Float64;
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid
    y
    yMid

    # problem definitions
    problem::String;

    # physical parameters
    sigmaT::SparseMatrixCSC{Float64,Int};
    sigmaS::SparseMatrixCSC{Float64,Int};
    Q::Array{Float64,1};

    # rank
    r::Int;
    epsAdapt::Float64;

    function settings(Nx::Int=302,Ny::Int=302,r::Int=15,problem::String="Linesource")

        # spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;

        if problem == "Linesource"
            a = -3.0; # left boundary
            b = 3.0; # right boundary

            c = -3.0; # lower boundary
            d = 3.0; # upper boundary

            Tend = 1.0;
            cfl = 0.7#1.7 # CFL condition
        
        elseif problem == "Lattice"
            a = 0.0; # left boundary
            b = 7.0; # right boundary

            c = 0.0; # lower boundary
            d = 7.0; # upper boundary

            Tend = 3.2;
            cfl = 0.4#0.5 # CFL condition
        else
            Throw(ArgumentError("Problem not defined"))
        end

        # spatial grid
        x = collect(range(a,stop = b,length = NCellsX));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx
        y = collect(range(c,stop = d,length = NCellsY));
        dy = y[2]-y[1];
        y = [y[1]-dy;y]; # add ghost cells so that boundary cell centers lie on a and b
        y = y.+dy/2;
        yMid = y[1:(end-1)].+0.5*dy
        

        nx,ny = NCellsX,NCellsY;
        Q = zeros(nx*ny);

        # physical parameters
        if problem == "Linesource"
            sigmaS = Diagonal(ones(Float64,nx*ny));
            sigmaA = Diagonal(zeros(Float64,nx*ny));
        elseif problem == "Lattice"
            sigmaS = ones(nx*ny);
            sigmaA = zeros(nx*ny);
            for i = 1:nx
                for j = 1:ny
                    if (xMid[i] <= 2.0 && xMid[i] >= 1.0) || (xMid[i] <= 6.0 && xMid[i] >= 5.0)
                        if (yMid[j] <= 2.0 && yMid[j] >= 1.0) || (yMid[j] <= 4.0 && yMid[j] >= 3.0) || (yMid[j] <= 6.0 && yMid[j] >= 5.0)
                            sigmaS[vectorIndex(ny,i,j)] = 0.0;
                            sigmaA[vectorIndex(ny,i,j)] = 10.0;
                        end
                    end 
                    if (xMid[i] <= 3.0 && xMid[i] >= 2.0) || (xMid[i] <= 5.0 && xMid[i] >= 4.0)
                        if (yMid[j] <= 3.0 && yMid[j] >= 2.0) || (yMid[j] <= 5.0 && yMid[j] >= 4.0)
                            sigmaS[vectorIndex(ny,i,j)] = 0.0;
                            sigmaA[vectorIndex(ny,i,j)] = 10.0;
                        end
                    end 
                    if xMid[i] <= 4.0 && xMid[i] >= 3.0
                        if yMid[j] <= 6.0 && yMid[j] >= 5.0
                            sigmaS[vectorIndex(ny,i,j)] = 0.0;
                            sigmaA[vectorIndex(ny,i,j)] = 10.0;
                        elseif yMid[j] <= 4.0 && yMid[j] >= 3.0
                            sigmaS[vectorIndex(ny,i,j)] = 0.0;
                            sigmaA[vectorIndex(ny,i,j)] = 10.0;
                            Q[vectorIndex(ny,i,j)] = 1.0;
                        end
                    end 
                end
            end
            sigmaA = Diagonal(sigmaA);
            sigmaS = Diagonal(sigmaS);
        end

        sigmaT = sparse(sigmaA .+ sigmaS);
        sigmaS = sparse(sigmaS);
       

        # time settings
        dt = cfl*dx;
        
        # number PN moments
        nPN = 39#39; # use odd number

        epsAdapt = 5e-2;

        # build class
        new(Nx,Ny,NCellsX,NCellsY,a,b,c,d,dx,dy,Tend,dt,cfl,nPN,x,xMid,y,yMid,problem,sigmaT,sigmaS,Q,r,epsAdapt);
    end
end

function IC(obj::settings,sample::Array{Float64,1},uncertParam::String)
    x = obj.xMid;
    y = obj.yMid;
    out = zeros(length(x),length(y));

    if obj.problem == "Linesource"
        if uncertParam == "0"
            x0 = sample[1];
            y0 = sample[2];
            s1 = 0.01
            s2 = 0.03^2
            floor = 1e-4
            for j = 1:length(x);
                for i = 1:length(y);
                    out[j,i] = max(floor,1.0/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0)+(y[i]-y0)*(y[i]-y0))/4.0/s2))/4.0/pi;
                end
            end
        elseif uncertParam == "1"
            x0 = 0.0;
            y0 = 0.0;
            s1 = 0.01
            s2 = 0.03^2
            floor = 1e-4
            for j = 1:length(x);
                for i = 1:length(y);
                    out[j,i] = max(floor,sample[1]/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0)+(y[i]-y0)*(y[i]-y0))/4.0/s2))/4.0/pi;
                end
            end
        elseif uncertParam == "2"
            x0 = sample[1];
            y0 = sample[2];
            s1 = 0.01
            s2 = 0.03^2
            floor = 1e-4
            for j = 1:length(x);
                for i = 1:length(y);
                    out[j,i] = max(floor,sample[3]/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0)+(y[i]-y0)*(y[i]-y0))/4.0/s2))/4.0/pi;
                end
            end
        else
            Throw(ArgumentError("Choose between 0 and 1 for the uncertParam as a String"))
        end
    elseif obj.problem == "Lattice"
        out = 1e-9*ones(length(x),length(y));
        nx = length(x);
        ny = length(y);
        if uncertParam == "0"
            sigmaA = obj.sigmaT - obj.sigmaS;
            for i = 1:nx
                for j = 1:ny
                    if (x[i] <= 2.0 && x[i] >= 1.0) || (x[i] <= 6.0 && x[i] >= 5.0)
                        if (y[j] <= 2.0 && y[j] >= 1.0) || (y[j] <= 4.0 && y[j] >= 3.0) || (y[j] <= 6.0 && y[j] >= 5.0)
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[1];
                        end
                    end 
                    if (x[i] <= 3.0 && x[i] >= 2.0) || (x[i] <= 5.0 && x[i] >= 4.0)
                        if (y[j] <= 3.0 && y[j] >= 2.0) || (y[j] <= 5.0 && y[j] >= 4.0)
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[2]; #
                        end
                    end 
                    if x[i] <= 4.0 && x[i] >= 3.0
                        if y[j] <= 6.0 && y[j] >= 5.0
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[3]; #
                        elseif y[j] <= 4.0 && y[j] >= 3.0
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[3]; # *sample[3]
                        end
                    end 
                end
            end
            obj.sigmaT = sigmaA .+ obj.sigmaS;
        elseif uncertParam == "1"   # source strength uncertainty: sample[1] ~ Uniform(1-p,1+p)
            Q_new = zeros(nx * ny)
            for i = 1:nx
                for j = 1:ny
                    if x[i] <= 4.0 && x[i] >= 3.0 && y[j] <= 4.0 && y[j] >= 3.0
                        Q_new[vectorIndex(ny, i, j)] = sample[1]
                    end
                end
            end
            obj.Q = Q_new
        elseif uncertParam == "2"   # absorption (sample[1..3]) + source (sample[4]) uncertainty
            sigmaA = obj.sigmaT - obj.sigmaS;
            Q_new  = zeros(nx * ny)
            for i = 1:nx
                for j = 1:ny
                    if (x[i] <= 2.0 && x[i] >= 1.0) || (x[i] <= 6.0 && x[i] >= 5.0)
                        if (y[j] <= 2.0 && y[j] >= 1.0) || (y[j] <= 4.0 && y[j] >= 3.0) || (y[j] <= 6.0 && y[j] >= 5.0)
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[1];
                        end
                    end
                    if (x[i] <= 3.0 && x[i] >= 2.0) || (x[i] <= 5.0 && x[i] >= 4.0)
                        if (y[j] <= 3.0 && y[j] >= 2.0) || (y[j] <= 5.0 && y[j] >= 4.0)
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[2];
                        end
                    end
                    if x[i] <= 4.0 && x[i] >= 3.0
                        if y[j] <= 6.0 && y[j] >= 5.0
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[1];
                        elseif y[j] <= 4.0 && y[j] >= 3.0
                            sigmaA[vectorIndex(ny,i,j)] = 10.0*sample[2];
                            Q_new[vectorIndex(ny,i,j)] = sample[3];
                        end
                    end
                end
            end
            obj.sigmaT = sigmaA .+ obj.sigmaS;
            obj.Q      = Q_new
        end
    end
    return out;
end