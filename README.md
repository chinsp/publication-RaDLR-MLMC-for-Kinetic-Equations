# RaDLR-MLMC for Kinetic Equations

This repository contains the Julia code accompanying the paper on rank-adaptive dynamic low-rank multilevel Monte Carlo (RaDLR-MLMC) methods for kinetic equations under uncertainty.

The core idea: solving kinetic equations (radiation transport, shallow water) is expensive. When input parameters are uncertain, you need many solves. This code reduces that cost in two ways:
1. **Low-rank solvers** — represent the solution as a product of small matrices, exploiting structure in the solution to reduce memory and compute.
2. **Multilevel Monte Carlo** — instead of running all samples on a fine grid, use a hierarchy of grids. Most samples run cheaply on coarse grids; only a few run on fine ones. The estimator remains accurate at a fraction of the cost.

---

## Repository structure

```
.
├── main.jl                          # Entry point — run studies from here
├── src/
│   ├── Integrators.jl               # Low-rank time integrators (augBUG, parBUG, etc.)
│   ├── uq.jl                        # Uncertainty quantification: MC, MLMC, control variates
│   └── plotting.jl                  # Publication-quality figures
├── scripts/
│   ├── runGaussianPulse_MC_MLMC_study.jl
│   ├── runLinesource_tol_study.jl
│   ├── runLattice_source_study.jl
│   └── runSW_param3_study.jl
└── test_problems/
    ├── radiation_transport/
    │   ├── 1D_slabgeometry/         # Solver for 1D problems (Gaussian Pulse, Linesource)
    │   └── 2D_slabgeometry/         # Solver for 2D problems (Lattice)
    └── shallow_water/               # Solver for shallow water shock problems
```

Output is written to `Results/` at the project root (created on first run).

---

## Test problems

| Flag | Problem | What it studies |
|---|---|---|
| `-gp` | **Gaussian Pulse** (1D radiation) | Compares plain Monte Carlo against MLMC across several tolerances |
| `-ls` | **Linesource** (2D radiation) | Tolerance study sweeping spatial and angular resolution |
| `-la` | **Lattice** (2D radiation, heterogeneous medium) | MLMC with uncertain absorption coefficients in inner/outer blocks |
| `-sw` | **Shallow Water** (1D shock) | MLMC over three uncertain initial-condition parameters simultaneously |

---

## Requirements

**Julia 1.9 or later.**

Install all packages from the Julia REPL:

```julia
using Pkg
Pkg.add([
    "PyPlot", "LaTeXStrings", "JLD2", "Distributions",
    "Interpolations", "GridInterpolations", "LinearRegression",
    "Parameters", "ProgressMeter", "ProgressBars",
    "FastGaussQuadrature", "LegendrePolynomials", "SpecialFunctions",
    "QuadGK", "SparseArrays", "SphericalHarmonicExpansions",
    "SphericalHarmonics", "TypedPolynomials", "GSL",
    "MultivariatePolynomials", "Einsum", "PyCall", "CUDA"
])
```

`PyPlot` requires a working Python installation with `matplotlib`:

```bash
pip install matplotlib
```

`CUDA` is only needed for the GPU-accelerated thermal radiation integrator. All other studies run on CPU.

---

## Running

All commands are run from the project root.

```bash
# Run a single study
julia main.jl --gaussian-pulse
julia main.jl --shallow-water

# Run multiple studies
julia main.jl --gaussian-pulse --lattice

# Run all studies in sequence
julia main.jl --all

# The Linesource study uses multithreading — pass --threads for a speedup
julia --threads auto main.jl --linesource

# Short-form flags
julia main.jl -gp -sw -la -ls
```

```
Flags:
  -gp, --gaussian-pulse
  -ls, --linesource
  -la, --lattice
  -sw, --shallow-water
  -a,  --all
  -h,  --help
```

Each study prints progress to stdout and saves figures and raw data (`.jld2`) under `Results/`.
