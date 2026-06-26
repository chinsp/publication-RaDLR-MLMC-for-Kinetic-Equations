include("settings.jl")
include("solver.jl")

using DelimitedFiles
using PyPlot, PyCall
using NPZ
using LegendrePolynomials
using GridInterpolations
# import PyCall: @pyimport
# @pyimport tikzplotlib as mpl
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

s = settings(problem = problem);
Solver = solver(s);
Solver.sample = [0.3]
Nnu = length(nu_list)
############################

tDLRA, uDLRA, alphaDLRA = SolveCorrectionFormulationRankAdaptBUG(Solver);

# %%

fig = figure()
ax = gca()
iq = 1
ax.plot(s.xMid, uDLRA[:,iq], linestyle=":",label="DLRA")
ax.set_title(L"h")
ax.tick_params("both")
ax.set_xlabel(L"$x$")
ax.legend()
fig.savefig("figures/$(problem)/compare-h_$(problem)_$(Solver.sample).pdf")

fig = figure()
ax = gca()
iq = 2
ax.plot(s.xMid, uDLRA[:,iq], linestyle=":",label="DLRA")
ax.set_title(L"$h u_m$")
ax.tick_params("both")
ax.set_xlabel(L"$x$")
ax.legend()
fig.savefig("figures/$(problem)/compare-hu_$(problem)_$(Solver.sample).pdf")


# %% plot of the profiles
N = s.N-2;
xpos = [0.05; 0.1;0.15; 0.3; 0.55; 0.65; 0.67]
idx = Int.(round.((xpos.-s.a).*s.NCells/(s.b-s.a)))
npos = length(xpos);
nH = 1000;
vDLRA = zeros(npos,nH)
hgridDLRA = zeros(npos,nH)
L0 = collectPl(-1.0, lmax = N);
for n = 1:npos
    hDLRA = uDLRA[idx[n],1];
    hgridDLRA[n,:] = collect(range(0,hDLRA,length=nH));
    for k = 1:nH
        L = collectPl(2*hgridDLRA[n,k]/hDLRA-1, lmax = N)./L0;

        tmp = uDLRA[idx[n],2]/hDLRA;
        for l = 1:N
            tmp += alphaDLRA[idx[n],l]*L[l]/hDLRA;
        end
        vDLRA[n,k] = tmp;

    end
end

# %%
fig = figure("v profiles slow",figsize=(16, 5), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
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

fig.savefig("figures/$(problem)/velocity_profile_close_$(problem)_$(Solver.sample).pdf")
