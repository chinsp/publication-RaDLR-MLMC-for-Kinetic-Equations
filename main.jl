# main.jl
#
# Entry point for the RaDLR-MLMC publication scripts.
# Run from the project root; output is written to Results/ under the project root.
#
# Usage:
#   julia main.jl [flags]
#
# Flags:
#   -gp, --gaussian-pulse    MC vs MLMC comparison for the Gaussian Pulse problem
#   -ls, --linesource        Tolerance study for the Linesource problem (spatial + angular)
#   -la, --lattice           MLMC source study for the Lattice problem
#   -sw, --shallow-water     MLMC combined-parameter study for the Shallow Water problem
#   -a,  --all               Run all four studies in sequence
#   -h,  --help              Print this message and exit
#
# Examples:
#   julia main.jl --all
#   julia main.jl --gaussian-pulse --shallow-water
#   julia main.jl -gp -sw
#   julia --threads auto main.jl --linesource

using Printf

const SCRIPTS_DIR = joinpath(@__DIR__, "scripts")

const HELP = """
Usage: julia main.jl [flags]

Flags:
  -gp, --gaussian-pulse    MC vs MLMC comparison for the Gaussian Pulse problem
  -ls, --linesource        Tolerance study for the Linesource problem (spatial + angular)
  -la, --lattice           MLMC heterogeneous-absorption source study for the Lattice problem
  -sw, --shallow-water     MLMC combined-parameter study for the Shallow Water problem
  -a,  --all               Run all four studies in sequence
  -h,  --help              Print this message and exit

Examples:
  julia main.jl --all
  julia main.jl --gaussian-pulse --shallow-water
  julia main.jl -gp -sw
  julia --threads auto main.jl --linesource   # Linesource uses Threads.@threads
"""

function parse_flags(args)
    if isempty(args) || any(a -> a in ("-h", "--help"), args)
        print(HELP)
        exit(isempty(args) ? 1 : 0)
    end

    run_gp = false
    run_ls = false
    run_la = false
    run_sw = false

    for arg in args
        if     arg in ("-gp", "--gaussian-pulse");  run_gp = true
        elseif arg in ("-ls", "--linesource");       run_ls = true
        elseif arg in ("-la", "--lattice");          run_la = true
        elseif arg in ("-sw", "--shallow-water");    run_sw = true
        elseif arg in ("-a",  "--all");              run_gp = run_ls = run_la = run_sw = true
        else
            println(stderr, "Unknown flag: $arg")
            print(stderr, HELP)
            exit(1)
        end
    end

    return (gp = run_gp, ls = run_ls, la = run_la, sw = run_sw)
end

function run_study(name::String, script::String)
    println()
    println("=" ^ 60)
    println("  $name")
    println("=" ^ 60)
    t = @elapsed include(script)
    @printf("\nFinished in %.1f s\n", t)
    println("-" ^ 60)
end

# ── entry point ────────────────────────────────────────────────

flags = parse_flags(ARGS)

flags.gp && run_study(
    "Gaussian Pulse — MC vs MLMC comparison",
    joinpath(SCRIPTS_DIR, "runGaussianPulse_MC_MLMC_study.jl"))

flags.ls && run_study(
    "Linesource — tolerance study (spatial + angular)",
    joinpath(SCRIPTS_DIR, "runLinesource_tol_study.jl"))

flags.la && run_study(
    "Lattice — heterogeneous-absorption MLMC source study",
    joinpath(SCRIPTS_DIR, "runLattice_source_study.jl"))

flags.sw && run_study(
    "Shallow Water — combined-parameter MLMC study",
    joinpath(SCRIPTS_DIR, "runSW_param3_study.jl"))
