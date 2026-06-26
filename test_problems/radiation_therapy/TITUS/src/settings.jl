__precompile__

using Images, FileIO, TOML, Meshes, MAT, DICOM

mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    Nz::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    NCellsZ::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    e::Float64;
    f::Float64;
    # grid cell width
    dx::Float64
    dy::Float64
    dz::Float64

    # time settings
    # end time
    mu_e
    eMax
    eMin
    eRest
    # time increment
    dE::Float64;
    # CFL number 
    cfl::Float64;
    #number beam energies
    N_E
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid
    y
    yMid
    z
    zMid

    # problem definitions
    problem::String;

    #particle type
    particle::String;
    # beam properties
    x0 #::Array{Float64,1};
    y0 #::Array{Float64,1};
    z0 #::Array{Float64,1};
    Omega1 #::Array{Float64,1};
    Omega2 #::Array{Float64,1};
    Omega3 #::Array{Float64,1};
    OmegaMin::Float64;
    densityMin::Float64;
    sigmaX::Float64; # spatial std of initial beam
    sigmaY::Float64; # spatial std of initial beam
    sigmaZ::Float64; # spatial std of initial beam
    sigmaE; # energy std of boundary beam

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,3};
    densityHU::Array{Float64,3};
    CT::Array{Float64,3}; #to store a finer resolution CT needed for later refinement

    # rank
    r::Int; #for adaptive this is initial rank
    rMax::Int; #and this is the max allowed rank

    gridSize::Array{Int,1};
    gridWidth::Array{Float64,1};
    gridScale::Array{Float64,1};; #scale factor of grid refinement relative to that given from an initial CT
    cropCT::Bool

    sizeOfTracerCT::Array{Int,1};

    # tolerance for rank adaptivity
    epsAdapt::Float64;  
    adaptIndex::Float64;

    #file names
    solverName::String;
    tracerFileName::String;
    model::String;

    #refinement related parameters
    indicator::Int32  # 1 - none, 2 - discrete CT/density, 3 - probability density
    max_depth::Int32  #  maximum refinement level
    metric::Int32     # 1 - gradient of dose est. 2- value of dose est. 3 - combination of gradient and value (for more indicators see mesh_refinement.f90)
    strictness::Float64 # how strict to be when choosing cells to refine in [0,1] 

    function Settings(filePath::String)
        # load config
        config = TOML.parsefile(filePath)
        #physics
        particle = get(config["physics"], "particle", "Protons")
        problem = get(config["physics"], "problem", "BoxInsert")
        model = get(config["physics"], "model", "Boltzmann")
        OmegaMin = get(config["physics"], "OmegaMin", 0)
        mu_e = get(config["physics"], "eKin", 90)

        #numerics
        Nx = get(config["numerics"], "nx", 43)
        Ny = get(config["numerics"], "ny", 43)
        Nz = get(config["numerics"], "nz", 163)
        r = get(config["numerics"], "rank", 20)
        rMax = get(config["numerics"], "maxRank", 100)
        order = get(config["numerics"], "order", 2)
        nPN = get(config["numerics"], "nMoments", 95)
        solverName = get(config["numerics"], "solverName", "Tracer_rankAdaptiveInEnergy")
        tracerFileName = get(config["numerics"], "tracerFileName", "eDep_$(problem)_$(model).bin")
        cfl = get(config["numerics"], "cfl", 10)
        epsAdapt = get(config["numerics"], "tolerance", 0.01)
        sizeOfTracerCT = get(config["numerics"], "sizeOfTracerCT",[1,1,1])

        #planning
        dataFile = get(config["planning"], "dataFile", "data/TG119.mat")
        Omega1 = get(config["planning"], "Omega1", 0.0)
        Omega2 = get(config["planning"], "Omega2", 0.0)
        Omega3 = get(config["planning"], "Omega3", 1.0)
        x0 = get(config["planning"], "x0", 1.0)
        y0 = get(config["planning"], "y0", 1.0)
        z0 = get(config["planning"], "z0", 0.0)
        gridScale = get(config["planning"], "gridScale", [1.0, 1.0, 1.0])

        #refinement
        indicator = get(config["refinement"], "indicator", 1)
        max_depth = get(config["refinement"], "max_depth", 1)
        metric = get(config["refinement"], "metric", 1)
        strictness = get(config["refinement"], "strict", 0.95)

        #Proton rest energy
        if particle == "Protons"
            eRest = 938.26 #MeV
        elseif particle == "Electrons"
            eRest = 0.5 #MeV -> not used here
            println("Only protons supported in this version of TITUS!")
        end
        # spatial grid setting
        if order ==1
            NCellsX = Nx - 1;
            NCellsY = Ny - 1;
            NCellsZ = Nz - 1;
        elseif order == 2
            NCellsX = Nx - 3;
            NCellsY = Ny - 3;
            NCellsZ = Nz - 3;
        end

        a = 0.0; # left boundary
        b = 2.0; # right boundary

        c = 0.0; # lower boundary
        d = 2.0; # upper boundary

        e = 0.0; # left z boundary
        f = 7.0; # right z boundary

        density = ones(NCellsX,NCellsY,NCellsZ); 
        densityHU = zeros(NCellsX,NCellsY,NCellsZ); #HU

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0;
        eMax = 90.0;
        densityMin = 0.2;
        adaptIndex = 1;
        sigmaX = 0.1;
        sigmaY = 0.1;
        sigmaZ = 0.01;
        eMin = 0.001;
        nE_perBeam = [0; 1]
        gridSize = [NCellsX,NCellsY,NCellsZ];
        cropCT = true;
        CT = zeros(0,0,0);

        if problem == "BoxInsert"
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 2.0; # upper boundary
            e = 0.0;
            f = 7.0;
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = 0.0;
            Omega2 = 0.0;
            Omega3 = 1.0;
            x0 = 0.5 * b;
            y0 = 0.5 * d;
            z0 = 0.0 * f;
            # density[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= 0.6190303991130821; #inserted box of lower density 
            densityHU[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= 1000; #inserted box of lower density defined in HU
            density = reshape(HUtoDensity(densityHU[:]),NCellsX,NCellsY,NCellsZ)
            sizeOfTracerCT = [NCellsX,NCellsY,NCellsZ]
        elseif problem == "TwoBeams"
            nB=2;
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 4.0; # upper boundary
            e = 0.0;
            f = 4.0;
            w_e = 1
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = maximum(mu_e) + 5*maximum(sigmaE); 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = zeros(nB)
            Omega2 = zeros(nB)
            Omega3 = zeros(nB)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            Omega1[1] = 0.0;
            Omega2[1] = 0.0;
            Omega3[1] = 1.0;

            # #90°
            Omega1[2] = 0.0;
            Omega2[2] = -1.0;
            Omega3[2] = 0.0;

            x0[1] = 0.5 * b;
            y0[1] = 0.5 * d;
            z0[1] = 0.0 * f;

            # #for 90°/60°
            x0[2] = 0.5 * b;
            y0[2] = 0.0 * d;
            z0[2] = 0.5 * f;
        elseif problem == "SingleBeam"
            nB=1;
            a = 0.0; # left boundary
            b = 3.0; # right boundary
            c = 0.0; # lower boundary
            d = 3.0; # upper boundary
            e = 0.0;
            f = 7.0;
            #mu_e = 50;
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; #maximum energy mean plus five standard devs
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = zeros(nB)
            Omega2 = zeros(nB)
            Omega3 = zeros(nB)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)

            Omega1[1] = 0.0;
            Omega2[1] = 0.0;
            Omega3[1] = 1.0;

            x0[1] = 0.5 * b;
            y0[1] = 0.5 * d;
            z0[1] = 0.0 * f;
            sizeOfTracerCT = [NCellsX,NCellsY,NCellsZ]
            CT = densityHU
        elseif problem == "dicomImport"
            nB = size(Omega1)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            #read dicom file
            densityHU, res = load_ct_volume(dataFile)
            NCellsX = size(densityHU,1)
            NCellsY = size(densityHU,2)
            NCellsZ = size(densityHU,3)

            dx = res[1]/10#/2  # divide by 10 bc of unit conversion mm -> cm, divide by 4 to scale down CT 
            dy = res[2]/10#/2 
            dz = res[3]/10#/2 
            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;
            density = reshape(HUtoDensity(densityHU[:]),NCellsX,NCellsY,NCellsZ)
            
            println("Resolution before trimming: dx = $dx, dy = $dy, dz = $dz, Bounds before trimming: [xmin, xmax] = [$a,$b], [ymin, ymax] = [$c,$d], [zmin, zmax] = [$e,$f]")
            w_e = [1, 1];
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = maximum(eKin) + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin));
            sigmaZ = 0.01; 
            x0[1] = 0.55 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.55 * f;
            x0[2] = 0.55 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.55 * f;
            #crop away air at the boundaries and regions far away from beam #
            densityHU, idx=trim_density(Int.(round.(densityHU)),eps=0.1,beams=[(SVector(x0[1], y0[1],z0[1]), SVector(2*sigmaX,2*sigmaY,sigmaE[1]), 100.0,SVector(Omega1[1], Omega2[1], Omega3[1])),(SVector(x0[2], y0[2],z0[2]), SVector(2*sigmaX,2*sigmaY,sigmaE[2]), 100.0, SVector(Omega1[2], Omega2[2], Omega3[2]))],x_range=(a,b), y_range=(c,d), z_range=(e,f))
            density = density[idx[1],idx[2],idx[3]]
            NCellsX = size(density,1)
            NCellsY = size(density,2)
            NCellsZ = size(density,3)
            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary, divide by 10 bc of unit conversion mm -> cm
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;
            # println("Positions before trimming: Beam 1: [$(x0[1]),$(y0[1]),$(z0[1])], Beam 2: [$(x0[2]),$(y0[2]),$(z0[2])]")
            #add option for finer grid for dose comp (maybe also only for DLRA)
            if gridScale[1] != 1.0 || gridScale[2] != 1.0  || gridScale[3] != 1.0 
                gridCTtoDose = interpolate((collect(range(a,b,size(density,1))),collect(range(c,d,size(density,2))),collect(range(e,f,size(density,3)))), density,Gridded(Linear()))
                gridCTtoDoseHU = interpolate((collect(range(a,b,size(density,1))),collect(range(c,d,size(density,2))),collect(range(e,f,size(density,3)))), densityHU,Gridded(Linear()))
            
                density = gridCTtoDose(collect(range(a,b,Int(round(size(density,1)*gridScale[1])))),collect(range(c,d,Int(round(size(density,2)*gridScale[2])))),collect(range(e,f,Int(round(size(density,3)*gridScale[3])))))
                densityHU = gridCTtoDoseHU(collect(range(a,b,Int(round(size(densityHU,1)*gridScale[1])))),collect(range(c,d,Int(round(size(densityHU,2)*gridScale[2])))),collect(range(e,f,Int(round(size(densityHU,3)*gridScale[3])))))
                
                # NCellsX = size(density,1)
                # NCellsY = size(density,2)
                # NCellsZ = size(density,3)
                # dx = dx/gridScale[1]
                # dy = dy/gridScale[2]
                # dz = dz/gridScale[3]
                NCellsX, NCellsY, NCellsZ = size(density)

                dx = (b - a) / (NCellsX - 1)
                dy = (d - c) / (NCellsY - 1)
                dz = (f - e) / (NCellsZ - 1)
            else
                NCellsX = size(density,1)
                NCellsY = size(density,2)
                NCellsZ = size(density,3)
                density = density[idx[1],idx[2],idx[3]]
                densityHU = densityHU[idx[1],idx[2],idx[3]]
            end

            # NCellsX = size(density,1)
            # NCellsY = size(density,2)
            # NCellsZ = size(density,3)
            # a = 0.0; # left boundary
            # b = NCellsX * dx; # right boundary, divide by 10 bc of unit conversion mm -> cm
            # c = 0.0; # lower boundary
            # d = NCellsY * dy; # upper boundary
            # e = 0.0;
            # f = NCellsZ * dz;
            println("Resolution after trimming: dx = $dx, dy = $dy, dz = $dz, Bounds after trimming: [xmin, xmax] = [$a,$b], [ymin, ymax] = [$c,$d], [zmin, zmax] = [$e,$f]")
            if order ==1
                Nx = NCellsX + 1;
                Ny = NCellsY + 1;
                Nz = NCellsZ + 1;
            elseif order == 2
                Nx = NCellsX + 3;
                Ny = NCellsY + 3;
                Nz = NCellsZ + 3;
            end

            sizeOfTracerCT = [NCellsX,NCellsY,NCellsZ]
            
            println("Size of tracer CT = $(sizeOfTracerCT)")
            println("Size of CT grid after trimming and rescaling: [$NCellsX,$NCellsY,$NCellsZ] Cells, dx= $dx. dy=$dy, dz=$dz")
            #redefine beam position because of changed grid, this probably needs to be done smarter (actually find equivalent position in changed grid or keep beam pos and change box borders (a,b),...)
            x0[1] = 0.5 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.5 * f;
            x0[2] = 0.5 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.5 * f;
            # println("Positions after trimming: Beam 1: [$(x0[1]),$(y0[1]),$(z0[1])], Beam 2: [$(x0[2]),$(y0[2]),$(z0[2])]")
            density[density.<0.0271] .= 0.0271
            densityHU[densityHU.<-999.6] .= -999.6

             #smoothen CT
            σ = 0.75  
            kernels = KernelFactors.gaussian((σ, σ, σ))
            densityHU = Int64.(round.(imfilter(densityHU, kernels)))
            density = imfilter(density, kernels)

            #write out trimmed dicom (for MC reference computation)
            save_cropped_ct_as_dicom(densityHU,
                                     dataFile,       # path to original DICOM slices
                                     dx, dy, dz,
                                     "cropped_dicom_output")
            #reduce to only one beam
            # x0 = 0.5 * b;
            # y0 = 0.0 * d;
            # z0 = 0.5 * f;
            # x0 = 0.5 * b;
            # y0 = 1.0 * d;
            # z0 = 0.5 * f;
            # Omega1 = Omega1[1]
            # Omega2 = Omega2[1]
            # Omega3= Omega3[1]
            # Omega1 = Omega1[2]
            # Omega2 = Omega2[2]
            # Omega3 = Omega3[2]
            # nB = 1
            #Q is a limit for minimum density needed for stability?
        elseif problem == "smallCT"
            Omega1 = [0.0,0.0]
            Omega2 = [1.0,-1.0]
            Omega3 = [0.0,0.0]

            nB = size(Omega1)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)

            img = load("data/2d_ct.png") #load CT slice
            gry = Float32.(Gray.(img))
            hu_2d = imresize(gry, (64, 64)).*400 .-160 #scale to HU range
            
            NX0, NY0 = size(hu_2d)   # 32 x 64
            NZ0       = 12            # base z-slices (≈ 6 cm slab at 5mm spacing)
            
            # Extrude into 3D: (NY, NX, NZ) = (32, 64, 12)
            hu_3d = repeat(reshape(hu_2d, NX0, NY0, 1), 1, 1, NZ0) #since we only look at small z-area we assume the slices to not vary too much
            
            # ── Physical bounds (cm) ─────────────────────────────────────────────────────
            a = 0.0
            b = 8.0
            c = 0.0
            d = 8.0
            e = 0.0
            f = 1.5

            x0[1] = 0.225 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.5 * f;
            x0[2] = 0.225 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.5 * f;

            w_e = [1, 1];
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = maximum(eKin) + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin));
            sigmaZ = 0.01; 

            densityHU = Int.(round.(resample_hu(hu_3d, NCellsX, NCellsY, NCellsZ,X_cm=b,Y_cm=d,Z_cm=f)))
            density = reshape(HUtoDensity(Float64.(densityHU[:])),NCellsX,NCellsY,NCellsZ)
            CT = Float64.(hu_3d)
            sizeOfTracerCT = [size(hu_3d,1),size(hu_3d,2),size(hu_3d,3)]
        elseif problem == "matImport"
            nB = size(Omega1)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            #read matfile (expected to be in the style as matRad phantoms)
            file = matopen(dataFile, "r")
            tmp = read(file, "ct") 
            # spatial grid setting
            NCellsX = tmp["cubeDim"][1]
            NCellsY = tmp["cubeDim"][2]
            NCellsZ = tmp["cubeDim"][3]

            dx = tmp["resolution"]["x"]/10 # divide by 10 bc of unit conversion mm -> cm
            dy = tmp["resolution"]["y"]/10
            dz = tmp["resolution"]["z"]/10
            
            density =  tmp["cube"][1]
            densityHU =  tmp["cubeHU"][1]

            #cut off bench and air around patient for PROSTATE.mat case
            density = density[Int(round(3*size(density,1)/10)):Int(round(6.5*size(density,1)/10)),Int(round(1.5*size(density,2)/10)):Int(round(8*size(density,2)/10)),Int(round(2.0*size(density,3)/10)):Int(round(8*size(density,3)/10))]
            densityHU = densityHU[Int(round(3*size(densityHU,1)/10)):Int(round(6.5*size(densityHU,1)/10)),Int(round(1.5*size(densityHU,2)/10)):Int(round(8*size(densityHU,2)/10)),Int(round(2.0*size(densityHU,3)/10)):Int(round(8*size(densityHU,3)/10))]
            NCellsX = Int(round(size(density,1)))
            NCellsY = Int(round(size(density,2)))
            NCellsZ = Int(round(size(density,3)))

            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;

            tmp = nothing
            close(file)
            w_e = [1, 1];
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = maximum(eKin) + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin));
            sigmaZ = 0.01; 

            x0[1] = 0.55 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.55 * f;
            x0[2] = 0.55 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.55 * f;

            # #crop away air at the boundaries and regions far away from beam 
            # densityHU_cropped, idx=trim_density(Int.(round.(densityHU)),eps=0.1,beams=[(SVector(x0[1], y0[1],z0[1]), SVector(3*sigmaX,3*sigmaY,sigmaE[1]), 100.0,SVector(Omega1[1], Omega2[1], Omega3[1])),(SVector(x0[2], y0[2],z0[2]), SVector(3*sigmaX,3*sigmaY,sigmaE[2]), 100.0, SVector(Omega1[2], Omega2[2], Omega3[2]))],x_range=(a,b), y_range=(c,d), z_range=(e,f))
            # density_cropped = density[idx[1],idx[2],idx[3]]
            densityHU_cropped = densityHU
            density_cropped = density
            # #add option for finer grid for dose comp (maybe also only for DLRA)
            if gridScale[1] != 1.0 || gridScale[2] != 1.0  || gridScale[3] != 1.0 
                gridCTtoDose = interpolate((collect(range(a,b,size(density_cropped,1))),collect(range(c,d,size(density_cropped,2))),collect(range(e,f,size(density_cropped,3)))), density_cropped,Gridded(Linear()))
                gridCTtoDoseHU = interpolate((collect(range(a,b,size(density_cropped,1))),collect(range(c,d,size(density_cropped,2))),collect(range(e,f,size(density_cropped,3)))), densityHU_cropped,Gridded(Linear()))
            
                density = gridCTtoDose(collect(range(a,b,Int(round(size(density_cropped,1)*gridScale[1])))),collect(range(c,d,Int(round(size(density_cropped,2)*gridScale[2])))),collect(range(e,f,Int(round(size(density_cropped,3)*gridScale[3])))))
                densityHU = gridCTtoDoseHU(collect(range(a,b,Int(round(size(density_cropped,1)*gridScale[1])))),collect(range(c,d,Int(round(size(density_cropped,2)*gridScale[2])))),collect(range(e,f,Int(round(size(density_cropped,3)*gridScale[3])))))
                
                NCellsX = Int(round(size(density_cropped,1)*gridScale[1]))
                NCellsY = Int(round(size(density_cropped,2)*gridScale[2]))
                NCellsZ = Int(round(size(density_cropped,3)*gridScale[3]))
                dx = dx/gridScale[1]
                dy = dy/gridScale[2]
                dz = dz/gridScale[3]
            else
                # NCellsX = size(density_cropped,1)
                # NCellsY = size(density_cropped,2)
                # NCellsZ = size(density_cropped,3)
                # density = density[idx[1],idx[2],idx[3]]
                # densityHU = densityHU[idx[1],idx[2],idx[3]]
            end
            println("Size of CT after cropping and rescaling = [$NCellsX, $NCellsY, $NCellsZ]")
            densityHU = round.(Int64,(densityHU)) # make sure Hounsfield units are ints
            density[density.<0.0271] .= 0.0271
            densityHU[densityHU.<-999] .= -999

             #smoothen CT
            σ = 0.75  
            kernels = KernelFactors.gaussian((σ, σ, σ))
            densityHU = Int64.(round.(imfilter(densityHU, kernels)))
            density = imfilter(density, kernels)
            density = density[:,end:-1:1,:]
            densityHU = densityHU[:,end:-1:1,:]


            
            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary, divide by 10 bc of unit conversion mm -> cm
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;

            if order ==1
                Nx = NCellsX + 1;
                Ny = NCellsY + 1;
                Nz = NCellsZ + 1;
            elseif order == 2
                Nx = NCellsX + 3;
                Ny = NCellsY + 3;
                Nz = NCellsZ + 3;
            end

            sizeOfTracerCT = [NCellsX,NCellsY,NCellsZ]
            #redefine beam position because of changed grid, this probably needs to be done smarter (actually find equivalent position in changed grid or keep beam pos and change box borders (a,b),...)
            # x0[1] = 0.5 * b;
            # y0[1] = 0.0 * d;
            # z0[1] = 0.5 * f;
            # x0[2] = 0.5 * b;
            # y0[2] = 1.0 * d;
            # z0[2] = 0.5 * f;

            #reduce to only one beam
            # x0 = 0.5 * b;
            # y0 = 0.0 * d;
            # z0 = 0.5 * f;
            # x0 = 0.5 * b;
            # y0 = 1.0 * d;
            # z0 = 0.5 * f;
            # Omega1 = Omega1[1]
            # Omega2 = Omega2[1]
            # Omega3= Omega3[1]
            # Omega1 = Omega1[2]
            # Omega2 = Omega2[2]
            # Omega3 = Omega3[2]
            # nB = 1
            #Q is a limit for minimum density needed for stability?
        else
            println("Problem $(problem) undefined.")
        end
        sigmaT = sigmaA + sigmaS;

        if order == 2
            # Initialize the grid with ghost cells for second-order accuracy
            x = collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];

            # Add two ghost cells on each boundary
            x = [x[1] - 2*dx; x[1] - dx; x; x[end] + dx]
            y = [y[1] - 2*dy; y[1] - dy; y; y[end] + dy]
            z = [z[1] - 2*dz; z[1] - dz; z; z[end] + dz]

            # Calculate the cell boundaries by shifting by half a grid spacing
            x = x .+ dx/2
            y = y .+ dy/2
            z = z .+ dz/2

            # Calculate the midpoints of the cells
            xMid = x[2:(end-2)] .+ 0.5 * dx
            yMid = y[2:(end-2)] .+ 0.5 * dy
            zMid = z[2:(end-2)] .+ 0.5 * dz
        else
            x = collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];
            x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
            x = x.+dx/2;
            xMid = x[1:(end-1)].+0.5*dx
            y = collect(range(c,stop = d,length = NCellsY));
            y = [y[1]-dy;y]; # add ghost cells so that boundary cell centers lie on a and b
            y = y.+dy/2;
            yMid = y[1:(end-1)].+0.5*dy
            z = collect(range(e,stop = f,length = NCellsZ));
            z = [z[1]-dz;z]; # add ghost cells so that boundary cell centers lie on a and b
            z = z.+dz/2;
            zMid = z[1:(end-1)].+0.5*dz
        end

        gridWidth = [dx,dy,dz]; 
        # time settings
        dE = cfl*min(dx,dy,dz)

        #sigmaE = maximum(sigmaE)
        # build class
  
        new(Nx,Ny,Nz,NCellsX,NCellsY,NCellsZ,a,b,c,d,e,f,dx,dy,dz,mu_e,eMax,eMin,eRest,dE,cfl,N_E,nPN,x,xMid,y,yMid,z,zMid,problem,particle,x0,y0,z0,Omega1,Omega2,Omega3,OmegaMin,densityMin,sigmaX,sigmaY,sigmaZ,sigmaE,sigmaT,sigmaS,density,densityHU,CT,r,rMax,gridSize,gridWidth,gridScale,cropCT,sizeOfTracerCT,epsAdapt,adaptIndex,solverName,tracerFileName,model,indicator,max_depth,metric,strictness);
    end
end
