using PyPlot
using LinearAlgebra
using JLD2
using PyCall
include("utils.jl")
ticker = pyimport("matplotlib.ticker")

testCase = "WB"
if testCase == "TwoBeams"
    dims = [2,4,4]
    numCells = [80,160,160]
    numCells_low = [20,40,40]
    theta = 0.01
    FP = true
else
    dims = [2,2,7]
    numCells = [80,80,280]
    numCells_low = [20,20,70]
    theta = 0.01
    FP = true
end

#load results 
if testCase == "WB"
    m_low = 75
    m_high = 115
    psi = Array{Float64}(undef,Int.(stat("../../topas/examples/Pia/RefCalcMC_protons_Edep_WB90MeV.bin").size/8))
    io_psi = open("../../topas/examples/Pia/RefCalcMC_protons_Edep_WB90MeV.bin", "r")
    read!(io_psi,psi);
    normFact = 1e8
    beamEn = 90
    doseMC = reshape(psi./normFact,numCells[1],numCells[2],numCells[3])
    doseMC = doseMC./sum(doseMC[:]).*beamEn
    
    dosePn = load("output/dose_fullPn_nPN$(m_low)_Boltzmann_E90_res1mm.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
    doseDLRA_low = load("output/dose_nPN$(m_low)_tol0.01_Boltzmann_E90_res1mm.jld2")["dose"]./4^3 
    doseDLRA_high = load("output/dose_nPN$(m_high)_tol0.01_Boltzmann_E90.jld2")["dose"]
    en_low = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E90_res1mm.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
    en_high = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E90.jld2")["energy"].-938.26
    rank_lowNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E90_res1mm.jld2")["rank"]
    rank_lowNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_Boltzmann_E90_res1mm.jld2")["rank"]
    rank_highNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E90.jld2")["rank"]
    rank_highNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_Boltzmann_E90.jld2")["rank"]
elseif testCase =="BI"
    m_low = 75
    m_high = 115
    psi = Array{Float64}(undef,Int.(stat("../../topas/examples/Pia/RefCalcMC_protons_Edep_BoxInsert_80MeV.bin").size/8))
    io_psi = open("../../topas/examples/Pia/RefCalcMC_protons_Edep_BoxInsert_80MeV.bin", "r")
    read!(io_psi,psi);
    normFact = 1
    beamEn = 80
    doseMC = reshape(psi./normFact,numCells[1],numCells[2],numCells[3])
    doseMC = doseMC./sum(doseMC[:]).*beamEn

    dosePn = load("output/dose_fullPn_nPN$(m_low)_Boltzmann_E80_res1mm_BoxInsert.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
    doseDLRA_low = load("output/dose_nPN$(m_low)_tol0.01_Boltzmann_E80_res1mm_BoxInsert.jld2")["dose"]./4^3
    doseDLRA_high = load("output/dose_nPN$(m_high)_tol0.01_eDep_BoxInsert_Boltzmann_highres.jld2")["dose"]
    en_low = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E80_res1mm_BoxInsert.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
    en_high = load("output/rankInEnergy_nPN$(m_low)_tol0.01_eDep_BoxInsert_Boltzmann_highres.jld2")["energy"].-938.26
    rank_lowNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_Boltzmann_E80_res1mm_BoxInsert.jld2")["rank"]
    rank_lowNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_Boltzmann_E80_res1mm_BoxInsert.jld2")["rank"]
    rank_highNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_eDep_BoxInsert_Boltzmann_highres.jld2")["rank"]
    rank_highNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_eDep_BoxInsert_Boltzmann_highres.jld2")["rank"]
elseif testCase == "TwoBeams"
    m_low = 75
    m_high = 115
    psi = Array{Float64}(undef,Int.(stat("../../topas/examples/Pia/RefCalcMC_protons_Edep_twoBeams_90deg.bin").size/8))
    io_psi = open("../../topas/examples/Pia/RefCalcMC_protons_Edep_twoBeams_90deg.bin", "r")
    read!(io_psi,psi);
    normFact = 1
    beamEn = 50
    doseMC = reshape(psi./normFact,numCells[1],numCells[2],numCells[3])
    doseMC = doseMC./sum(doseMC[:]).*beamEn

    dosePn_FP = load("output/dose_fullPn_nPN$(m_low_FP)_eDep_TwoBeams_Boltzmann_90.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
    doseDLRA_low = load("output/dose_nPN$(m_low)_tol0.01_eDep_TwoBeams_Boltzmann_90.jld2")["dose"]./4^3
    doseDLRA_high = load("output/dose_nPN$(m_high)_tol0.01_eDep_TwoBeams_Boltzmann_90_highres.jld2")["dose"]
    en_low = load("output/rankInEnergy_nPN$(m_low)_tol0.01_eDep_TwoBeams_Boltzmann_90.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
    en_high = load("output/rankInEnergy_nPN$(m_high)_tol0.01_eDep_TwoBeams_Boltzmann_90_highres.jld2")["energy"].-938.26
    rank_lowNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_eDep_TwoBeams_Boltzmann_90.jld2")["rank"]
    rank_lowNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_eDep_TwoBeams_Boltzmann_90.jld2")["rank"]
    rank_highNlowM = load("output/rankInEnergy_nPN$(m_low)_tol0.01_eDep_TwoBeams_Boltzmann_90_highres.jld2")["rank"]
    rank_highNhighM = load("output/rankInEnergy_nPN$(m_high)_tol0.01_eDep_TwoBeams_Boltzmann_90_highres.jld2")["rank"]
end

if FP
    if testCase == "WB"
        m_low_FP = 19
        m_high_FP = 75
        dosePn_FP = load("output/dose_fullPn_nPN$(m_low_FP)_FP_E90_res1mm_corrected.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
        doseDLRA_low_FP = load("output/dose_nPN$(m_low_FP)_tol0.01_FP_E90_res1mm_corrected.jld2")["dose"]./4^3 
        doseDLRA_high_FP = load("output/dose_nPN$(m_high_FP)_tol0.01_FP_E90_corrected.jld2")["dose"] 
        en_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E90_res1mm_corrected.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
        en_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E90_res1mm_corrected.jld2")["energy"].-938.26
        en_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E90_corrected.jld2")["energy"].-938.26
        en_highNhighM_FP =load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E90_corrected.jld2")["energy"].-938.26
        rank_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E90_res1mm_corrected.jld2")["rank"]
        rank_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E90_res1mm_corrected.jld2")["rank"]
        rank_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E90_corrected.jld2")["rank"]
        rank_highNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E90_corrected.jld2")["rank"]
    elseif testCase =="BI"
        m_low_FP = 19
        m_high_FP = 75
        dosePn_FP = load("output/dose_fullPn_nPN$(m_low_FP)_FP_E80_BoxInsert_res1mm_corrected.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
        doseDLRA_low_FP = load("output/dose_nPN$(m_low_FP)_tol0.01_FP_E80_BoxInsert_res1mm_corrected.jld2")["dose"]./4^3 
        doseDLRA_high_FP = load("output/dose_nPN$(m_high_FP)_tol0.01_FP_E80_BoxInsert_corrected.jld2")["dose"] 
        en_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E80_BoxInsert_res1mm_corrected.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
        en_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E80_BoxInsert_res1mm_corrected.jld2")["energy"].-938.26
        en_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E80_BoxInsert_corrected.jld2")["energy"].-938.26
        en_highNhighM_FP =load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E80_BoxInsert_corrected.jld2")["energy"].-938.26
        rank_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E80_BoxInsert_res1mm_corrected.jld2")["rank"]
        rank_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E80_BoxInsert_res1mm_corrected.jld2")["rank"]
        rank_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_FP_E80_BoxInsert_corrected.jld2")["rank"]
        rank_highNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_FP_E80_BoxInsert_corrected.jld2")["rank"]
    elseif testCase == "TwoBeams"
        m_low_FP = 19
        m_high_FP = 75
        dosePn_FP = load("output/dose_fullPn_nPN$(m_low_FP)_eDep_TwoBeams_FP_90.jld2")["dose"]./4^3 #has to be adjusted to different grid sizes (since we plot energy dep. per cell)
        doseDLRA_low_FP = load("output/dose_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90.jld2")["dose"]./4^3
        doseDLRA_high_FP = load("output/dose_nPN$(m_high_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["dose"]
        en_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90.jld2")["energy"].-938.26 #deduct rest energy to get kinetic
        en_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90.jld2")["energy"].-938.26
        en_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["energy"].-938.26
        en_highNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["energy"].-938.26
        en_highNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["energy"].-938.26
        rank_lowNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90.jld2")["rank"]
        rank_lowNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_eDep_TwoBeams_FP_90.jld2")["rank"]
        rank_highNlowM_FP = load("output/rankInEnergy_nPN$(m_low_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["rank"]
        rank_highNhighM_FP = load("output/rankInEnergy_nPN$(m_high_FP)_tol0.01_eDep_TwoBeams_FP_90_highres.jld2")["rank"]
    end
end
#define indices and grids for Plotting
xMid_MC = collect(range(0,dims[1],numCells[1]))
yMid_MC = collect(range(0,dims[2],numCells[2]))
zMid_MC = collect(range(0,dims[3],numCells[3]))

xMid_DLRA = collect(range(0,dims[1],numCells[1]))
yMid_DLRA = collect(range(0,dims[2],numCells[2]))
zMid_DLRA = collect(range(0,dims[3],numCells[3]))

xMid_low_DLRA = collect(range(0,dims[1],numCells_low[1]))
yMid_low_DLRA = collect(range(0,dims[2],numCells_low[2]))
zMid_low_DLRA = collect(range(0,dims[3],numCells_low[3]))


xMid_low_Pn = collect(range(0,dims[1],numCells_low[1]))
yMid_low_Pn = collect(range(0,dims[2],numCells_low[2]))
zMid_low_Pn = collect(range(0,dims[3],numCells_low[3]))

X_MC = (xMid_MC'.*ones(size(yMid_MC)))
Y_MC = (yMid_MC'.*ones(size(xMid_MC)))'
Z_MC = (zMid_MC'.*ones(size(yMid_MC)))
XZ_MC = (xMid_MC'.*ones(size(zMid_MC)))
ZX_MC = (zMid_MC'.*ones(size(xMid_MC)))
YZ_MC = (yMid_MC'.*ones(size(zMid_MC)))'

X_DLRA = (xMid_DLRA'.*ones(size(yMid_DLRA)))
Y_DLRA = (yMid_DLRA'.*ones(size(xMid_DLRA)))'
Z_DLRA = (zMid_DLRA'.*ones(size(yMid_DLRA)))
XZ_DLRA = (xMid_DLRA'.*ones(size(zMid_DLRA)))
ZX_DLRA = (zMid_DLRA'.*ones(size(xMid_DLRA)))
YZ_DLRA = (yMid_DLRA'.*ones(size(zMid_DLRA)))'
    
X_low_DLRA = (xMid_low_DLRA'.*ones(size(yMid_low_DLRA)))
Y_low_DLRA = (yMid_low_DLRA'.*ones(size(xMid_low_DLRA)))'
Z_low_DLRA = (zMid_low_DLRA'.*ones(size(yMid_low_DLRA)))
XZ_low_DLRA = (xMid_low_DLRA'.*ones(size(zMid_low_DLRA)))
ZX_low_DLRA = (zMid_low_DLRA'.*ones(size(xMid_low_DLRA)))
YZ_low_DLRA = (yMid_low_DLRA'.*ones(size(zMid_low_DLRA)))'

X_low_Pn = (xMid_low_Pn'.*ones(size(yMid_low_Pn)))
Y_low_Pn = (yMid_low_Pn'.*ones(size(xMid_low_Pn)))'
Z_low_Pn = (zMid_low_Pn'.*ones(size(yMid_low_Pn)))
XZ_low_Pn = (xMid_low_Pn'.*ones(size(zMid_low_Pn)))
ZX_low_Pn = (zMid_low_Pn'.*ones(size(xMid_low_Pn)))
YZ_low_Pn = (yMid_low_Pn'.*ones(size(zMid_low_Pn)))'

#define x,y,z points where we want to cut 
if testCase == "TwoBeams"
    x = dims[1]/2
    y = dims[2]/2
    z1 = dims[3]/4*0.75
    z2 = dims[3]/4*1.5
    z3 = dims[3]/4*2.25
elseif testCase == "BI"
    x = dims[1]/2
    y = dims[2]/2
    z1 = dims[3]/4
    z2 = dims[3]/2
    z3 = dims[3]/7*5.5
else
    x = dims[1]/2
    y = dims[2]/2
    z1 = dims[3]/4
    z2 = dims[3]/2
    z3 = dims[3]/7*6
end 

idxX_DLRA = findmin(abs.(xMid_DLRA .- x))[2]
idxY_DLRA = findmin(abs.(yMid_DLRA .-y))[2] 
idxZ_1_DLRA = findmin(abs.(zMid_DLRA .- z1))[2]
idxZ_2_DLRA = findmin(abs.(zMid_DLRA .- z2))[2]
idxZ_3_DLRA = findmin(abs.(zMid_DLRA .- z3))[2]

idxX_low_DLRA = findmin(abs.(xMid_low_DLRA .- x))[2]
idxY_low_DLRA = findmin(abs.(yMid_low_DLRA .-y))[2] 
idxZ_low_DLRA_1 = findmin(abs.(zMid_low_DLRA .- z1))[2]
idxZ_low_DLRA_2 = findmin(abs.(zMid_low_DLRA .- z2))[2]
idxZ_low_DLRA_3 = findmin(abs.(zMid_low_DLRA .- z3))[2]

idxX_MC = findmin(abs.(xMid_MC .- x))[2]
idxY_MC = findmin(abs.(yMid_MC .-y))[2] 
idxZ_1_MC = findmin(abs.(zMid_MC .- z1))[2]
idxZ_2_MC = findmin(abs.(zMid_MC .- z2))[2]
idxZ_3_MC = findmin(abs.(zMid_MC .- z3))[2]

idxX_low_Pn = findmin(abs.(xMid_low_Pn .- x))[2]
idxY_low_Pn = findmin(abs.(yMid_low_Pn .-y))[2] 
idxZ_low_Pn_1 = findmin(abs.(zMid_low_Pn .- z1))[2]
idxZ_low_Pn_2 = findmin(abs.(zMid_low_Pn .- z2))[2]
idxZ_low_Pn_3 = findmin(abs.(zMid_low_Pn .- z3))[2]

prop_cycle = plt.rcParams["axes.prop_cycle"]
mycolors = prop_cycle.by_key()["color"]

#Plotting dose
close("all")
#Dose cuts 
#lateral
fig = figure("Dose, MC vs DLRA ϑ= 0.01 lat. cuts",dpi=500)
ax = gca()
ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_1_MC],linestyle="solid", label="z=$(round(z1,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,color=mycolors[1])
ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_2_MC],linestyle="solid", label="z=$(round(z2,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,marker="o",markevery=4,color=mycolors[1])
ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_3_MC],linestyle="solid", label="z=$(round(z3,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,marker="x",markevery=4,color=mycolors[1])
ax.plot(yMid_DLRA,doseDLRA_high[idxX_DLRA,:,idxZ_1_DLRA],linestyle="dashed", label="z=$(round(z1,digits=2)), DLRA",linewidth=2, alpha=1.0,color=mycolors[2])
ax.plot(yMid_DLRA,doseDLRA_high[idxX_DLRA,:,idxZ_2_DLRA],linestyle="dashed", label="z=$(round(z2,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="o",markevery=4,color=mycolors[2])
ax.plot(yMid_DLRA,doseDLRA_high[idxX_DLRA,:,idxZ_3_DLRA],linestyle="dashed", label="z=$(round(z3,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="x",markevery=4,color=mycolors[2])
ax.tick_params("both",labelsize=10) 
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("dep. energy [MeV]", fontsize=10)
ax.legend(loc="upper left", fontsize=10)
plt.legend(handlelength=3)
tight_layout()
savefig("JCP/LatCut_MCvsDLRA_$testCase.png")

fig = figure("Dose, Pn vs DLRA ϑ= 0.01 lat. cuts",dpi=500)
ax = gca()
ax.plot(yMid_low_Pn,dosePn[idxX_low_Pn,:,idxZ_low_Pn_1],linestyle="solid", label="z=$(round(z1,digits=2)), full rank",linewidth=2, alpha=1.0,color=mycolors[1])
ax.plot(yMid_low_Pn,dosePn[idxX_low_Pn,:,idxZ_low_Pn_2],linestyle="solid", label="z=$(round(z2,digits=2)), full rank",linewidth=2, alpha=1.0,marker="o",markevery=1,color=mycolors[1])
ax.plot(yMid_low_Pn,dosePn[idxX_low_Pn,:,idxZ_low_Pn_3],linestyle="solid", label="z=$(round(z3,digits=2)), full rank",linewidth=2, alpha=1.0,marker="x",markevery=1,color=mycolors[1])
ax.plot(yMid_low_DLRA,doseDLRA_low[idxX_low_DLRA,:,idxZ_low_DLRA_1],linestyle="dashed", label="z=$(round(z1,digits=2)), DLRA",linewidth=2, alpha=1.0,color=mycolors[2])
ax.plot(yMid_low_DLRA,doseDLRA_low[idxX_low_DLRA,:,idxZ_low_DLRA_2],linestyle="dashed", label="z=$(round(z2,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="o",markevery=1,color=mycolors[2])
ax.plot(yMid_low_DLRA,doseDLRA_low[idxX_low_DLRA,:,idxZ_low_DLRA_3],linestyle="dashed", label="z=$(round(z3,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="x",markevery=1,color=mycolors[2])
ax.tick_params("both",labelsize=10) 
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("dep. energy [MeV]", fontsize=10)
ax.legend(loc="upper left", fontsize=10)
plt.legend(handlelength=3)
tight_layout()
savefig("JCP/LatCut_PnvsDLRA_$testCase.png")

# #depth
fig = figure("Dose, MC vs DLRA vs Pn ϑ= 0.01 lat. cuts",dpi=500)
ax = gca()
#average btw two middle cells to get line exactly on material boundary and be able to compare on diff. grids (for heterogeneous)
ax.plot(zMid_MC[1:end],(doseMC[idxX_MC,idxY_MC,1:end]+doseMC[idxX_MC,idxY_MC+1,1:end])./2,linestyle="solid", label="Monte Carlo",linewidth=2, alpha=1.0,color=mycolors[1])
ax.plot(zMid_DLRA,(doseDLRA_high[idxX_DLRA,idxY_DLRA,:]+doseDLRA_high[idxX_DLRA,idxY_DLRA+1,:])./2,linestyle="dashed", label="DLRA, nPn=$m_high, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[2])
ax.plot(zMid_low_DLRA,(doseDLRA_low[idxX_low_DLRA,idxY_low_DLRA,:]+doseDLRA_low[idxX_low_DLRA,idxY_low_DLRA-1,:])./2,linestyle="dotted", label="DLRA, nPn=$m_low, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[3])
ax.plot(zMid_low_Pn,(dosePn[idxX_low_Pn,idxY_low_Pn,:]+dosePn[idxX_low_Pn,idxY_low_Pn-1,:])./2,linestyle="dashdot", label="full rank, nPn=$m_low, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[4])
ax.tick_params("both",labelsize=10) 
plt.xlabel("z [cm]", fontsize=10)
plt.ylabel("dep. energy [MeV]", fontsize=10)
ax.legend(loc="upper left", fontsize=10)
plt.legend(handlelength=3)
tight_layout()
savefig("JCP/DepthCut_MCvsDLRAvsPn_$testCase.png")

if FP
    fig = figure("Dose, MC vs DLRA ϑ= 0.01 lat. cuts, FP",dpi=500)
    ax = gca()
    ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_1_MC],linestyle="solid", label="z=$(round(z1,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,color=mycolors[1])
    ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_2_MC],linestyle="solid", label="z=$(round(z2,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,marker="o",markevery=4,color=mycolors[1])
    ax.plot(yMid_MC,doseMC[idxX_MC,:,idxZ_3_MC],linestyle="solid", label="z=$(round(z3,digits=2)), Monte Carlo",linewidth=2, alpha=1.0,marker="x",markevery=4,color=mycolors[1])
    ax.plot(yMid_DLRA,doseDLRA_high_FP[idxX_DLRA,:,idxZ_1_DLRA],linestyle="dashed", label="z=$(round(z1,digits=2)), DLRA",linewidth=2, alpha=1.0,color=mycolors[2])
    ax.plot(yMid_DLRA,doseDLRA_high_FP[idxX_DLRA,:,idxZ_2_DLRA],linestyle="dashed", label="z=$(round(z2,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="o",markevery=4,color=mycolors[2])
    ax.plot(yMid_DLRA,doseDLRA_high_FP[idxX_DLRA,:,idxZ_3_DLRA],linestyle="dashed", label="z=$(round(z3,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="x",markevery=4,color=mycolors[2])
    ax.tick_params("both",labelsize=10) 
    plt.xlabel("y [cm]", fontsize=10)
    plt.ylabel("dep. energy [MeV]", fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    plt.legend(handlelength=3)
    tight_layout()
    savefig("JCP/LatCut_MCvsDLRA_FP_$testCase.png")

    fig = figure("Dose, Pn vs DLRA ϑ= 0.01 lat. cuts, FP",dpi=500)
    ax = gca()
    ax.plot(yMid_low_Pn,dosePn_FP[idxX_low_Pn,:,idxZ_low_Pn_1],linestyle="solid", label="z=$(round(z1,digits=2)), full rank",linewidth=2, alpha=1.0,color=mycolors[1])
    ax.plot(yMid_low_Pn,dosePn_FP[idxX_low_Pn,:,idxZ_low_Pn_2],linestyle="solid", label="z=$(round(z2,digits=2)), full rank",linewidth=2, alpha=1.0,marker="o",markevery=1,color=mycolors[1])
    ax.plot(yMid_low_Pn,dosePn_FP[idxX_low_Pn,:,idxZ_low_Pn_3],linestyle="solid", label="z=$(round(z3,digits=2)), full rank",linewidth=2, alpha=1.0,marker="x",markevery=1,color=mycolors[1])
    ax.plot(yMid_low_DLRA,doseDLRA_low_FP[idxX_low_DLRA,:,idxZ_low_DLRA_1],linestyle="dashed", label="z=$(round(z1,digits=2)), DLRA",linewidth=2, alpha=1.0,color=mycolors[2])
    ax.plot(yMid_low_DLRA,doseDLRA_low_FP[idxX_low_DLRA,:,idxZ_low_DLRA_2],linestyle="dashed", label="z=$(round(z2,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="o",markevery=1,color=mycolors[2])
    ax.plot(yMid_low_DLRA,doseDLRA_low_FP[idxX_low_DLRA,:,idxZ_low_DLRA_3],linestyle="dashed", label="z=$(round(z3,digits=2)), DLRA",linewidth=2, alpha=1.0,marker="x",markevery=1,color=mycolors[2])
    ax.tick_params("both",labelsize=10) 
    plt.xlabel("y [cm]", fontsize=10)
    plt.ylabel("dep. energy [MeV]", fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    plt.legend(handlelength=3)
    tight_layout()
    savefig("JCP/LatCut_PnvsDLRA_FP_$testCase.png")


    #depth
    fig = figure("Dose, MC vs DLRA vs Pn ϑ= 0.01 lat. cuts, FP",dpi=500)
    ax = gca()
    #average btw two middle cells to get line exactly on material boundary and be able to compare on diff. grids (for heterogeneous)
    ax.plot(zMid_MC[1:end-1],(doseMC[idxX_MC,idxY_MC,2:end]+doseMC[idxX_MC,idxY_MC+1,2:end])./2,linestyle="solid", label="Monte Carlo",linewidth=2, alpha=1.0,color=mycolors[1])
    ax.plot(zMid_DLRA,(doseDLRA_high_FP[idxX_DLRA,idxY_DLRA,:]+doseDLRA_high_FP[idxX_DLRA,idxY_DLRA+1,:])./2,linestyle="dashed", label="DLRA, nPn=$m_high_FP, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[2])
    ax.plot(zMid_low_DLRA,(doseDLRA_low_FP[idxX_low_DLRA,idxY_low_DLRA,:]+doseDLRA_low_FP[idxX_low_DLRA,idxY_low_DLRA-1,:])./2,linestyle="dotted", label="DLRA, nPn=$m_low_FP, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[3])
    ax.plot(zMid_low_Pn,(dosePn_FP[idxX_low_Pn,idxY_low_Pn,:]+dosePn_FP[idxX_low_Pn,idxY_low_Pn-1,:])./2,linestyle="dashdot", label="full rank, nPn=$m_low_FP, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[4])
    ax.tick_params("both",labelsize=10) 
    plt.xlabel("z [cm]", fontsize=10)
    plt.ylabel("dep. energy [MeV]", fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    plt.legend(handlelength=3)
    tight_layout()
    savefig("JCP/DepthCut_MCvsDLRAvsPn_FP_$testCase.png")
end

#SurfPlots
fig = figure(figsize=(dims[2]+1.75,dims[3])) #choose figsize so plot had correct aspect ratio despite color bar
ax1 = gca()
im1 = ax1.pcolormesh(YZ_MC',Z_MC',doseMC[idxX_MC,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
cbar = plt.colorbar(im1,ax=ax1)
cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("z [cm]", fontsize=10)
# # Set tick locations
if testCase == "TwoBeams"
ax1.set_yticks([3,4,z1,z2,z3])
elseif testCase == "BI"
    ax1.set_xticks([0,0.5,y,1.5,2])
    ax1.set_yticks([1,2,3,4,5,6,7,z1,z2,z3])
else
    ax1.set_xticks([0,0.5,y,1.5,2])
    ax1.set_yticks([1,2,3,4,5,7,z1,z2,z3])
end
 
# Draw vertical and horizontal lines to indicate cuts
x_cut = [xMid_MC[idxX_MC]]
y_cut1 = [zMid_MC[idxZ_1_MC]]
y_cut2 = [zMid_MC[idxZ_2_MC]]
y_cut3 = [zMid_MC[idxZ_3_MC]]

if testCase != "TwoBeams"
    ax1.axvline(x=x_cut, color="white", linestyle="--", label="X Cut")
end
ax1.axhline(y=y_cut1, color="white", linestyle="--", label="Y Cut1")
ax1.axhline(y=y_cut2, color="white", linestyle="--", label="Y Cut2")
ax1.axhline(y=y_cut3, color="white", linestyle="--", label="Y Cut3")
tight_layout()
savefig("JCP/SurfPlot_MC_$testCase.png")

fig = figure(figsize=(dims[2]+1.75,dims[3]))
ax1 = gca()
im1 = ax1.pcolormesh(YZ_DLRA',Z_DLRA',doseDLRA_high[idxX_DLRA,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
cbar = plt.colorbar(im1,ax=ax1)
if testCase != "TwoBeams"
ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
else
ax1.set_yticks([1.00,2.00,3.00,4.00])   
end
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("z [cm]", fontsize=10)
cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
tight_layout()
savefig("JCP/SurfPlot_DLRA_highres_$testCase.png")

fig = figure(figsize=(dims[2]+1.75,dims[3]))
ax1 = gca()
if testCase != "TwoBeams"
ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
else
ax1.set_yticks([1.00,2.00,3.00,4.00])   
end
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
im1 = ax1.pcolormesh(YZ_low_DLRA',Z_low_DLRA',doseDLRA_low[idxX_low_DLRA,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
cbar = plt.colorbar(im1,ax=ax1)
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("z [cm]", fontsize=10)
cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
tight_layout()
savefig("JCP/SurfPlot_DLRA_lowres_$testCase.png")

fig = figure(figsize=(dims[2]+1.75,dims[3]))
ax1 = gca()
if testCase != "TwoBeams"
ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
else
ax1.set_yticks([1.00,2.00,3.00,4.00])   
end
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
im1 = ax1.pcolormesh(YZ_low_Pn',Z_low_Pn',dosePn[idxX_low_Pn,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
cbar = plt.colorbar(im1,ax=ax1)
plt.xlabel("y [cm]", fontsize=10)
plt.ylabel("z [cm]", fontsize=10)
cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
tight_layout()
savefig("JCP/SurfPlot_fullrank_$testCase.png")

if FP
    fig = figure(figsize=(dims[2]+1.75,dims[3]))
    ax1 = gca()
    if testCase != "TwoBeams"
    ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
    else
    ax1.set_yticks([1.00,2.00,3.00,4.00])   
    end
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    im1 = ax1.pcolormesh(YZ_DLRA',Z_DLRA',doseDLRA_high_FP[idxX_DLRA,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
    cbar = plt.colorbar(im1,ax=ax1)
    plt.xlabel("y [cm]", fontsize=10)
    plt.ylabel("z [cm]", fontsize=10)
    cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
    tight_layout()
    savefig("JCP/SurfPlot_DLRA_highres_FP_$testCase.png")

    fig = figure(figsize=(dims[2]+1.75,dims[3]))
    ax1 = gca()
    im1 = ax1.pcolormesh(YZ_low_DLRA',Z_low_DLRA',doseDLRA_low_FP[idxX_low_DLRA,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
    if testCase != "TwoBeams"
    ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
    else
    ax1.set_yticks([1.00,2.00,3.00,4.00])   
    end
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar = plt.colorbar(im1,ax=ax1)
    plt.xlabel("y [cm]", fontsize=10)
    plt.ylabel("z [cm]", fontsize=10)       
    cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
    tight_layout()
    savefig("JCP/SurfPlot_DLRA_lowres_FP_$testCase.png")

    fig = figure(figsize=(dims[2]+1.75,dims[3]))
    ax1 = gca()
    if testCase != "TwoBeams"
    ax1.set_yticks([1.00,2.00,3.00,4.00,5.00,6.00,7.00])
    else
    ax1.set_yticks([1.00,2.00,3.00,4.00])   
    end
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    im1 = ax1.pcolormesh(YZ_low_Pn',Z_low_Pn',dosePn_FP[idxX_low_Pn,:,:]',vmin=0,vmax=maximum(doseMC[idxX_MC,:,:]),cmap="jet")
    cbar = plt.colorbar(im1,ax=ax1)
    plt.xlabel("y [cm]", fontsize=10)
    plt.ylabel("z [cm]", fontsize=10)
    cbar.set_label("dep. energy [MeV]", rotation=270,labelpad=15)
    tight_layout()
    savefig("JCP/SurfPlot_fullrank_FP_$testCase.png")
end

#ranks 
fig = figure("rank in energy",dpi=500)
ax = gca()
ax.plot(en_high[:],linestyle="solid",rank_highNhighM[2:end], label="nPn=$m_high, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[1])
ax.plot(en_high[:],linestyle="dashed",rank_highNlowM[2:end], label="nPn=$m_low, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[1])
ax.plot(en_low[:],linestyle="solid",rank_lowNhighM[2:end], label="nPn=$m_high, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[2])
ax.plot(en_low[:],linestyle= "dashed",rank_lowNlowM[2:end], label="nPn=$m_low, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[2])
ax.set_xlabel("energy [MeV]", fontsize=10);
ax.set_ylabel("rank", fontsize=10);
ax.tick_params("both",labelsize=10);
ax.legend(loc="upper left", fontsize=10)
plt.legend(handlelength=3)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("JCP/ranksInEnergy_$testCase.png")

if FP
    fig = figure("rank in energy,FP",dpi=500)
    ax = gca()
    ax.plot(en_highNhighM_FP[:],linestyle="solid",rank_highNhighM_FP[2:end], label="nPn=$m_high_FP, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[1])
    ax.plot(en_highNlowM_FP[:],linestyle="dashed",rank_highNlowM_FP[2:end], label="nPn=$m_low_FP, dx=0.25mm",linewidth=2, alpha=1.0,color=mycolors[1])
    ax.plot(en_lowNhighM_FP[:],linestyle="solid",rank_lowNhighM_FP[2:end], label="nPn=$m_high_FP, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[2])
    ax.plot(en_lowNlowM_FP[:],linestyle= "dashed",rank_lowNlowM_FP[2:end], label="nPn=$m_low_FP, dx=1mm",linewidth=2, alpha=1.0,color=mycolors[2])
    ax.set_xlabel("energy [MeV]", fontsize=10);
    ax.set_ylabel("rank", fontsize=10);
    ax.tick_params("both",labelsize=10);
    ax.legend(loc="upper right", fontsize=10)
    plt.legend(loc="upper right",handlelength=3)
    tight_layout()
    fig.canvas.draw() # Update the figure
    savefig("JCP/ranksInEnergy_FP_$testCase.png")
end
