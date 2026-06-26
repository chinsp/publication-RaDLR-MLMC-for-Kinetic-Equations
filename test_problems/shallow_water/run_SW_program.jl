include("settings.jl")
include("solver.jl")

using DelimitedFiles
using PyPlot, PyCall
using NPZ
using LegendrePolynomials
using GridInterpolations
import PyCall: @pyimport
@pyimport tikzplotlib as mpl
# to load mpl do the following:
# import Pkg; Pkg.add("Conda")
# using Conda
# Conda.add("tikzplotlib",channel="conda-forge")
# %%
close("all")

problem = "shock"
#problem = "KowalskiTorrihon"
if !isdir("figures/")
    mkdir("figures");
end

if !isdir("figures/$(problem)")
    mkdir("figures/$(problem)");
end

if problem=="shock"
    nu_list = [1e1,1e-1]
    nu_test = 1
else
    nu_list = [1e-1,1e-2]
    nu_test = 0.1
end

s = Settings(problem = problem);
Nnu = length(nu_list)
############################
function solve_multiple(nu_list,s)
    solvers=[]
    local q_hist_tot = 0
    local qdot_hist_tot = 0
    tcpu_fom = 0
    for (i,nu) in enumerate(nu_list)
        s.nu = nu
        solver = DLRSolver(s);
        tcpu_fom += @elapsed tEnd, uhist,alpha_hist = SolveCorrectionFormulation_hist(solver);
        q_hist = cat(uhist,alpha_hist,dims=2)
        if i == 1
            q_hist_tot = q_hist
            qdot_hist_tot = (q_hist[:,:,2:end]-q_hist[:,:,1:end-1])/s.dt
        else
            q_hist_tot = cat(q_hist_tot,q_hist,dims=3)
            qdot_hist_tot = cat(q_hist_tot,(q_hist[:,:,2:end]-q_hist[:,:,1:end-1])/s.dt,dims=3)
        end
        push!(solvers,solver)
    end
    return q_hist_tot,qdot_hist_tot, solvers, tcpu_fom
end

q_hist_tot, qdot_hist_tot, solvers, tcpu_offline = solve_multiple(nu_list,s)
##
fig = figure(219)
ax = gca()
ax.pcolormesh(q_hist_tot[:,1,:])
ax.set_ylabel(L"space samples $n_x$")
ax.set_xlabel(L"time samples $n_t$")
ax.set_xticks([0,401,802])
ax.set_yticks([])
ax.text(10,10,L"$\nu=10$",color="red")
ax.text(410,10,L"$\nu=0.1$",color="red")
ax.plot([401,401], [0,1001], "r-.")
mpl.save("figures/$(problem)/snaphotset_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")

#%%
#@time tEnd, uFull = Solve(solver); alpha = uFull[:,3:end]; u = uFull[:,1:2];
#@time tEnd, uSW,alphaSW, uhistSW = SolveSW_hist(solver);
#@time tEnd, uDLRA, alphaDLRA = SolveCorrectionFormulationUnconventional(solver);
Ngrid,N,Ns = size(q_hist_tot)
tcpu_svd = @elapsed U,S,W = svd(reshape(permutedims(q_hist_tot[:,3:end,:],(1,3,2)),Ngrid*Ns,N-2),full=false)


# %% plot singular values

fig = figure(16)
ax = gca()
rel_err = ones(size(S)[1]+1)
rel_err[2:end] = ones(size(S))-cumsum(S.^2)/sum(S.^2)

ax.semilogy(rel_err,"*")
ax.set_xlim([-0.5,15.5])
ax.set_yticks(10.0 .^(-range(0,stop=13)))
ax.set_ylim([1e-12,ax.get_ylim()[2]])
#ax.minorticks_on()
s.nu = nu_test;
#solver = DLRSolver(s);
#ax.tcpu_rom_DLRA = @elapsed tEnd, uDLRA, alphaDLRA = SolveCorrectionFormulationUnconventional(solver);
ax.yaxis.set_tick_params(which="minor", right = "off")
ax.grid(which="minor",axis="y",linestyle="--")
ax.grid(b=true,which="major")
ax.set_xlabel("number of modes")
ax.set_ylabel("relative online error")
mpl.save("figures/$(problem)/POD-rel_error_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")
#%%
s.nu = nu_test
solver = DLRSolver(s);
s2 = Settings(2*s.Nx; problem = problem);
solver2 = DLRSolver(s2);
tcpu_fom_HSWE = @elapsed tEnd, uhist,alpha_hist = SolveCorrectionFormulation_hist(solver);

tcpu_fom_HSWE2 = @elapsed tEnd2, uhist2,alpha2_hist = SolveCorrectionFormulation_hist(solver2);
tcpu_rom_POD = @elapsed t, uMOR,alphaMOR = SolveCorrectionFormulationUnconventional_MOR(solver, W);
tcpu_fom_SWE = @elapsed tEnd, uSW,alphaSW = SolveSW(solver);
tcpu_rom_DLRA = @elapsed tEnd, uDLRA, alphaDLRA = SolveCorrectionFormulationUnconventional(solver);

# %%

fig = figure()
ax = gca()
iq = 1
h=ax.pcolormesh(uhist[:,iq,:])
ax.set_title(L"water height $h$")
ax.tick_params("both")
ax.set_ylabel(L"$x$")
ax.set_xlabel(L"$t$")
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(h, ax=ax)
mpl.save("figures/$(problem)/height-xt_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")


fig = figure()
ax = gca()
iq = 1
ax.plot(s.xMid,uhist[:,iq,end],linestyle="-", label="HSWME")
ax.plot(s.xMid,uSW[:,iq],linestyle="-.", label="SWE")
ax.plot(s.xMid, uMOR[:,iq,end],linestyle="--", label="POD-Galerkin")
ax.plot(s.xMid, uDLRA[:,iq], linestyle=":",label="DLRA")
ax.set_title(L"h")
ax.tick_params("both")
ax.set_xlabel(L"$x$")
ax.legend()
mpl.save("figures/$(problem)/compare-h_$(problem).tex",fig)

fig = figure()
ax = gca()
iq = 2
ax.plot(s.xMid, uhist[:,iq,end],linestyle="-", label="HSWME")
ax.plot(s.xMid, uSW[:,iq],linestyle="-.", label="SWE")
ax.plot(s.xMid, uMOR[:,iq,end],linestyle="--", label="POD-Galerkin")
ax.plot(s.xMid, uDLRA[:,iq], linestyle=":",label="DLRA")
ax.set_title(L"$h u_m$")
ax.tick_params("both")
ax.set_xlabel(L"$x$")
ax.legend()
mpl.save("figures/$(problem)/compare-hu_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")
rel_err_DLRA = norm(uhist[:,:,end]-uDLRA)/norm(uhist[:,:,end])
rel_err_MOR = norm(uhist[:,:,end]-uMOR[:,:,end])/norm(uhist[:,:,end])

println("")
println("rel err DLRA: ", rel_err_DLRA)
println("rel err MOR: ", rel_err_MOR)

# %% plot of the profiles
u = uhist[:,1:2,end]
alpha = alpha_hist[:,:,end]
N = s.N-2;
xpos = [0.05; 0.1;0.15; 0.3; 0.55; 0.65; 0.67]
idx = Int.(round.((xpos.-s.a).*s.NCells/(s.b-s.a)))
npos = length(xpos);
nH = 1000;
v = zeros(npos,nH)
vSW = zeros(npos,nH)
vDLRA = zeros(npos,nH)
vMOR = zeros(npos,nH)
hgrid = zeros(npos,nH)
hgridSW = zeros(npos,nH)
hgridDLRA = zeros(npos,nH)
hgridMOR = zeros(npos,nH)
L0 = collectPl(-1.0, lmax = N);
for n = 1:npos
    h = u[idx[n],1];
    hSW = uSW[idx[n],1];
    hMOR = uMOR[idx[n],1,end];
    hgrid[n,:] = collect(range(0,h,length=nH));
    hgridSW[n,:] = collect(range(0,hSW,length= nH));
    hgridMOR[n,:] = collect(range(0,hMOR,length= nH));

    hDLRA = uDLRA[idx[n],1];
    hgridDLRA[n,:] = collect(range(0,hDLRA,length=nH));
    for k = 1:nH
        L = collectPl(2*hgrid[n,k]/h-1, lmax = N)./L0;
        tmp = u[idx[n],2]/h;
        for l = 1:N
            tmp += alpha[idx[n],l]*L[l]/h;
        end
        v[n,k] = tmp;

        vSW[n,k] = uSW[idx[n],2]/hSW;

        tmp = uDLRA[idx[n],2]/hDLRA;
        for l = 1:N
            tmp += alphaDLRA[idx[n],l]*L[l]/hDLRA;
        end
        vDLRA[n,k] = tmp;

        tmp = uMOR[idx[n],2,end]/hMOR;
        for l = 1:N
            tmp += alphaMOR[idx[n],l,end]*L[l]/hMOR;
        end
        vMOR[n,k] = tmp;
    end
end

# %%
fig = figure("v profiles slow",figsize=(16, 5), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
# HSWE
ax.plot(v[1,:],hgrid[1,:], "b-", linewidth=2, label="\$x="*string(xpos[1])*"\$, HSWME", alpha=0.6)
ax.plot(v[2,:],hgrid[2,:], "r-", linewidth=2, label="\$x="*string(xpos[2])*"\$, HSWME", alpha=0.6)
ax.plot(v[3,:],hgrid[3,:], "k-", linewidth=2, label="\$x="*string(xpos[3])*"\$, HSWME", alpha=0.6)
# SW
if problem == "shock"
 ax.plot(vSW[1,:],hgridSW[1,:], "b--", linewidth=2, label="\$x="*string(xpos[1])*"\$, shallow water", alpha=0.6)
 ax.plot(vSW[2,:],hgridSW[2,:], "r--", linewidth=2, label="\$x="*string(xpos[2])*"\$, shallow water", alpha=0.6)
 ax.plot(vSW[3,:],hgridSW[3,:], "k--", linewidth=2, label="\$x="*string(xpos[3])*"\$, shallow water", alpha=0.6)
end
# MOR
ax.plot(vMOR[1,:],hgridMOR[1,:], "b-.", linewidth=2, label="\$x="*string(xpos[1])*"\$, POD-Galerkin", alpha=0.6)
ax.plot(vMOR[2,:],hgridMOR[2,:], "r-.", linewidth=2, label="\$x="*string(xpos[2])*"\$, POD-Galerkin", alpha=0.6)
ax.plot(vMOR[3,:],hgridMOR[3,:], "k-.", linewidth=2, label="\$x="*string(xpos[3])*"\$, POD-Galerkin", alpha=0.6)
# DLRA
ax.plot(vDLRA[1,:],hgridDLRA[1,:], "b:", linewidth=2, label="\$x="*string(xpos[1])*"\$, DLRA", alpha=1.0)
ax.plot(vDLRA[2,:],hgridDLRA[2,:], "r:", linewidth=2, label="\$x="*string(xpos[2])*"\$, DLRA", alpha=1.0)
ax.plot(vDLRA[3,:],hgridDLRA[3,:], "k:", linewidth=2, label="\$x="*string(xpos[3])*"\$, DLRA", alpha=1.0)
ax.legend(loc="upper left",fontsize=15)
#ax.set_xlim([0.0*minimum(v),1.1*maximum(v[1:3,:])])
ax.set_ylim([-0.01,1.01*maximum(hgridDLRA)])
ax.set_ylabel(L"$z$",fontsize=15)
ax.tick_params("both",labelsize=15)
ax.set_xlabel(L"$v$",fontsize=15)
show()

mpl.save("figures/$(problem)/velocity_profile_close_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")

#%%
fig = figure("v profiles fast",figsize=(16, 5), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
# MOR
ls_MOR = "--"
#HSWE
#ax.plot(v[4,:],hgrid[4,:], "b-", linewidth=2, label="\$x="*string(xpos[4])*"\$, HSWME", alpha=0.6)
#ax.plot(v[5,:],hgrid[5,:], "r-", linewidth=2, label="\$x="*string(xpos[5])*"\$, HSWME", alpha=0.4)
ax.plot(v[6,:],hgrid[6,:], "k-", linewidth=2, label="\$x="*string(xpos[6])*"\$, HSWME", alpha=0.4)
ax.plot(v[7,:],hgrid[7,:], "g-", linewidth=2, label="\$x="*string(xpos[7])*"\$, HSWME", alpha=0.4)
# SWE
if problem == "shock"
    #ax.plot(vSW[5,:],hgridSW[5,:], "r-.", linewidth=2, label="\$x="*string(xpos[5])*"\$, shallow water", alpha=0.6)
    ax.plot(vSW[6,:],hgridSW[6,:], "k-.", linewidth=2, label="\$x="*string(xpos[6])*"\$, shallow water", alpha=0.6)
    ax.plot(vSW[7,:],hgridSW[7,:], "g-.", linewidth=2, label="\$x="*string(xpos[7])*"\$, shallow water", alpha=0.6)
end
# POD Galerkin
#ax.plot(vMOR[5,:],hgridMOR[5,:], "r"*ls_MOR, linewidth=2, label="\$x="*string(xpos[5])*"\$, POD-Galerkin", alpha=0.6)
ax.plot(vMOR[6,:],hgridMOR[6,:], "k"*ls_MOR, linewidth=2, label="\$x="*string(xpos[6])*"\$, POD-Galerkin", alpha=0.6)
ax.plot(vMOR[7,:],hgridMOR[7,:], "g"*ls_MOR, linewidth=2, label="\$x="*string(xpos[7])*"\$, POD-Galerkin", alpha=0.6)
# DLRA
#ax.plot(vDLRA[4,:],hgridDLRA[4,:], "b:", linewidth=2, label="\$x="*string(xpos[4])*"\$, DLRA", alpha=1.0)
#ax.plot(vDLRA[5,:],hgridDLRA[5,:], "r:", linewidth=2, label="\$x="*string(xpos[5])*"\$, DLRA", alpha=1.0)
ax.plot(vDLRA[6,:],hgridDLRA[6,:], "k:", linewidth=2, label="\$x="*string(xpos[6])*"\$, DLRA", alpha=1.0)
ax.plot(vDLRA[7,:],hgridDLRA[7,:], "g:", linewidth=2, label="\$x="*string(xpos[7])*"\$, DLRA", alpha=1.0)
ax.legend(loc="upper left",fontsize=15)
#ax.set_xlim([0.15,1.1*maximum(v[4:6,:])])
ax.set_ylim([-0.01,1.01*maximum(hgridDLRA)])
ax.set_ylabel(L"$z$",fontsize=15)
ax.tick_params("both",labelsize=15)
ax.set_xlabel(L"$v$",fontsize=15)
show()

mpl.save("figures/$(problem)/velocity_profile_far_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")


# %%
max_rank = 15
ranks = range(1,stop=max_rank)
rel_err_MOR = zeros(length(ranks))
rel_err_DLRA = zeros(length(ranks))
speedup_POD = zeros(length(ranks))
speedup_DLRA = zeros(length(ranks))
for (ir,rank) in enumerate(ranks)
    s.r = rank;
    solver = DLRSolver(s);
    # POD
    speedup_POD[ir] = @elapsed t, uMOR,alphaMOR_hist = SolveCorrectionFormulationUnconventional_MOR(solver, W);
    speedup_POD[ir] /= tcpu_fom_HSWE
    rel_err_MOR[ir] = norm(uhist[:,:,end]-uMOR[:,:,end])/norm(uhist[:,:,end])
    # DLRA
    speedup_DLRA[ir] = @elapsed tEnd, uDLRA, alphaDLRA = SolveCorrectionFormulationUnconventional(solver);
    speedup_DLRA[ir] /= tcpu_fom_HSWE
    rel_err_DLRA[ir] = norm(uhist[:,:,end]-uDLRA)/norm(uhist[:,:,end])
    println("rank: ", rank)
    println("err POD: ", rel_err_MOR[ir],"  DLRA: ", rel_err_DLRA[ir])
    println("speedup POD: ", speedup_POD[ir],"  DLRA :", speedup_DLRA[ir])
end
speedup_DLRA = speedup_DLRA.^(-1)
speedup_POD = speedup_POD.^(-1)
# %%
fig = figure(6)
ax = gca()
ax.semilogy(rel_err,"*",label=L"offline $\mathcal{E}_\mathrm{off}$")
ax.semilogy(ranks,rel_err_MOR,marker="*",linestyle="",label=L"online $\mathcal{E}_\mathrm{on}$")
ax.set_xlabel("number of modes")
ax.set_ylabel("relative error")
ax.yaxis.set_tick_params(which="minor", right = "off")
ax.grid(which="minor",axis="y",linestyle=":")
ax.grid(b=true,which="major")
ax.set_xlim([-0.5,15.5])
ax.set_ylim([1e-13,1])
ax.legend()
mpl.save("figures/$(problem)/POD-rel_online-offline_error_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")


# %%
fig = figure()
ax = gca()
width = 15

#ax.set_yscale("log")
ax.grid()
ax.grid(which="minor",axis="y",linestyle=":")
ax.bar(00,tcpu_fom_HSWE,width=width)
ax.bar(20,tcpu_rom_DLRA,width=width)
ax.bar(40,tcpu_fom_SWE,width=width)
ax.bar(55,tcpu_rom_POD,width=5)
ax.bar(65,tcpu_svd,width=5,bottom=0)
ax.bar(65,tcpu_offline,width=5,bottom=tcpu_svd)
ax.set_ylabel(L"$t_\mathrm{cpu}$ [sec]")
ax.set_xticks([00,20,40,60])
ax.set_xticklabels(["HSWE","DLRA","SWE","POD-Galerkin"])

for (i, v) in enumerate([tcpu_fom_HSWE,tcpu_rom_DLRA,tcpu_fom_SWE])
    ax.text((i-1)* 20, v+3, string(round(v*10)/10," s"), color="blue", fontweight="bold", ha="center")
end
ax.text(55, tcpu_rom_POD+3, string(round(tcpu_rom_POD*10)/10," s"), color="blue", fontweight="bold", ha="center")
ax.text(65, tcpu_svd+tcpu_offline+3, string(round((tcpu_svd+tcpu_offline)*10)/10," s"), color="blue", fontweight="bold", ha="center")
ax.text(55, 2*tcpu_offline/5, "online costs",  fontweight="bold", ha ="center",rotation=90)
ax.text(65, 2*tcpu_offline/5, "offline costs (SVD + FOM)",  fontweight="bold",ha ="center", rotation=90)

mpl.save("figures/$(problem)/costs_comparison_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")
#%% error vs speedup
close("all")
fig = figure()
ax = gca()
iq = 2
ax.plot(rel_err_MOR, speedup_POD,linestyle="",marker="o", label="POD-Galerkin")
ax.plot(rel_err_DLRA,speedup_DLRA,linestyle="",marker="+", label="DLRA")
#ax.set_yscale("log")
ax.set_xscale("log")
#ax.set_xlabel(L"$\frac{\Vert \mathbf{u}(x,t=0.02)-\tilde{\mathbf{u}}(x,t=0.02)\Vert_2}{\Vert \mathbf{u}(x,t=0.02) \Vert_2}$")
ax.set_xlabel("rel err.")
ax.set_ylabel("speedup")
ax.legend()
ax.grid()
ax.grid(which="minor",axis="y",linestyle=":")


for ir in ranks[1:end]
    ax.text(rel_err_MOR[ir], speedup_POD[ir]+2,string(ranks[ir]), ha="center")
end
#ax.plot([(1,1.5), (2,0.5)], arrow = arrow(:closed, 0.1), color = :blue)


for ir in [1,2,3,5,6,7,9,8,14]
    ax.text(rel_err_DLRA[ir], speedup_DLRA[ir]+2,string(ranks[ir]),ha="center")
end

mpl.save("figures/$(problem)/speedup_vs_err_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")


# %% compare with double different discretization
max_rank = 8
ranks = range(1,stop=max_rank)
rel_err_MOR = zeros(length(ranks))
rel_err_DLRA = zeros(length(ranks))
speedup_POD = zeros(length(ranks))
speedup_DLRA = zeros(length(ranks))
uref = uhist2[1:2:end-1,:,end]
for (ir,rank) in enumerate(ranks)
    s.r = rank;
    solver = DLRSolver(s);
    # POD
    speedup_POD[ir] = @elapsed t, uMOR,alphaMOR_hist = SolveCorrectionFormulationUnconventional_MOR(solver, W);
    speedup_POD[ir] /= tcpu_fom_HSWE
    rel_err_MOR[ir] = norm(uref-uMOR[:,:,end])/norm(uref)
    # DLRA
    speedup_DLRA[ir] = @elapsed tEnd, uDLRA, alphaDLRA = SolveCorrectionFormulationUnconventional(solver);
    speedup_DLRA[ir] /= tcpu_fom_HSWE
    rel_err_DLRA[ir] = norm(uref-uDLRA)/norm(uref)
    println("rank: ", rank)
    println("err POD: ", rel_err_MOR[ir],"  DLRA: ", rel_err_DLRA[ir])
    println("speedup POD: ", speedup_POD[ir],"  DLRA :", speedup_DLRA[ir])
end
speedup_DLRA = speedup_DLRA.^(-1)
speedup_POD = speedup_POD.^(-1)

#%% error vs speedup
close("all")
fig = figure()
ax = gca()
iq = 2
ax.plot(rel_err_MOR, speedup_POD,linestyle="",marker="o", label="POD-Galerkin")
ax.plot(rel_err_DLRA,speedup_DLRA,linestyle="",marker="+", label="DLRA")
#ax.set_yscale("log")
#ax.set_xlabel(L"$\frac{\Vert \mathbf{u}(x,t=0.02)-\tilde{\mathbf{u}}(x,t=0.02)\Vert_2}{\Vert \mathbf{u}(x,t=0.02) \Vert_2}$")
ax.set_xlabel("rel err.")
ax.set_ylabel("speedup")
ax.legend()
ax.grid()
ax.grid(which="minor",axis="y",linestyle=":")


for ir in ranks[1:end]
    ax.text(rel_err_MOR[ir], speedup_POD[ir]+2,string(ranks[ir]), ha="center")
end
#ax.plot([(1,1.5), (2,0.5)], arrow = arrow(:closed, 0.1), color = :blue)


for ir in [1,2,3,5,6,7,8]
    ax.text(rel_err_DLRA[ir], speedup_DLRA[ir]+2,string(ranks[ir]),ha="center")
end

mpl.save("figures/$(problem)/speedup_vs_err_$(problem)_dblprec.tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")

#%% plot conservation

s.r = 3;
nalpha = 4;
solver = DLRSolver(s);
for i in [1,2,3]
    if i ==1
        t,u,a = SolveCorrectionFormulation_hist(solver);
        mylabel = "HSWME"
        mystyle= "-"
    elseif i==2
        t,u,a = SolveCorrectionFormulationUnconventional_MOR(solver,W);
        mylabel = "POD-Galerkin"
        mystyle = ":"
    else
        times, u, a = SolveCorrectionFormulationUnconventional_hist(solver);
        mylabel = "DLRA"
        mystyle = "--"
        t = times[end]
    end
    #%%
    Ntime = size(u,3)
    times = collect(range(0,stop=t,step=s.dt))
    h0 = sum(u[2:end-1,1,2])
    hu0 = sum(u[2:end-1,2,2])
    a0 = sum(a[2:end-1,nalpha,2])


    dH = zeros(Ntime)
    dHU = zeros(Ntime)
    dA = zeros(Ntime)
    for it in range(2,stop=Ntime)
        dH[it] = h0 - sum(u[2:end-1,1,it])
        dHU[it] = hu0 - sum(u[2:end-1,2,it])
        dA[it] = a0 - sum(a[2:end-1,nalpha,it])
    end
    dH *=s.dx
    dHU *=s.dx
    dA *=s.dx
    #%%
    fig = figure(101)
    subplot(3,1,1)
    ax = gca()
    ax.plot(times,dH,mystyle)
    ax.set_ylabel("mass")
    ax.grid(which="major",axis="x",linestyle=":")
    ax.set_xticklabels([])
    subplot(3,1,2)
    ax = gca()
    ax.plot(times,dHU, mystyle,label = mylabel)
    ax.set_ylabel("momentum")
    ax.grid(which="major",axis="x",linestyle=":")
    ax.set_xticklabels([])
    ax.legend()
    subplot(3,1,3)
    ax = gca()
    ax.plot(times,dA,mystyle)
    ax.set_ylabel("higher momentum")
    ax.grid(which="major",axis="x",linestyle=":")
    #ax.plot(times,dHU,label=L"$\Delta hu$")
    ax.set_xlabel(L"time $t$")
end
ax.legend(["HSWME","POD-Galerkin","DLRA"],loc=1)
mpl.save("figures/$(problem)/conservation_$(problem).tex",fig,axis_height="\\figureheight",axis_width="\\figurewidth")
