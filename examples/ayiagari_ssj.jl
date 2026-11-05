# Aiyagari model, calibrated and dimensioned as the Krusell-Smith steady state
# of the sequence-space jacobian toolkit (Auclert, Bardóczy, Rognlie, Straub, 2021),
# `notebooks/krusell_smith.ipynb`:
#
#     eis=1, delta=0.025, alpha=0.11, rho_e=0.966, sd_e=0.5, L=1, nE=7, nA=500,
#     amin=0, amax=200;  targets: r=0.01, Y=1
#
# Differences:
# - the toolkit solves for β (with K and Z) to hit r=1%; here β is fixed to the
#   value that gives r=1% (found once, with this grid) and we solve for K;
# - the endogenous state is cash on hand y = (1+r)a + w e, on a log-spaced grid
#   (Dolo's `LogGrid`, i.e. the toolkit's `agrid` with pivot 0.25) in y instead of a;
# - the agent is solved by the endogenous grid method, then Newton on Dolo's
#   residuals (linear interpolation, as in the toolkit).
#
#     julia --project=. examples/ayiagari_ssj.jl

using Dolark, Dolo

include("ayiagari.jl")

α = 0.11
δ = 0.025
r = 0.01
ρ = 0.966
sd_e = 0.5           # cross-sectional standard deviation of log efficiency
L = 1.0

K_ss = α/(r + δ)     # capital consistent with the targets
A = K_ss^(-α)        # TFP such that Y = 1
w = (1 - α)*A*K_ss^α
σ = sd_e*sqrt(1 - ρ^2)    # Dolo's rouwenhorst takes the volatility of innovations

β = 0.9819518241     # gives r = 1% on this grid

n_e = 7
n_a = 500
a_max = 200.0

model = let
    m = ayiagari_model(; γ=1.0, ρ, σ, n_e, normalize_e=true, α, δ, A, N=L, K=3.0)
    e = [exp(q[3]) for q in m.agent.exogenous.Q]
    y_min = w*minimum(e)                     # a = 0
    y_max = (1 + r)*a_max + w*maximum(e)     # a = a_max
    ayiagari_model(; β, γ=1.0, ρ, σ, n_e, normalize_e=true, y_min, y_max, α, δ, A, N=L, K=3.0)
end

discretization = Dict(:endo => Dict(:y => Dolo.LogGrid(n_a; pivot=0.25)))

println(model)
println("grid: ", n_e, " × ", n_a, " = ", n_e*n_a, " points")

# compilation
solve_steady_state(model; discretization, verbose=false)
solve_steady_state_joint(model; discretization, verbose=false)
solve_steady_state_joint(model; discretization, linsolve=:gmres, agent_steps=5, verbose=false)

t = @elapsed sol = solve_steady_state(model; discretization, verbose=false)
println("\nnested Newton (exact jacobian):  ", round(t, digits=3), " s, ", sol.iterations, " iterations, K = ", sol.y.K)

t = @elapsed s = solve_steady_state_joint(model; discretization, verbose=false)
println("joint Newton (direct):           ", round(t, digits=3), " s, ", s.iterations, " iterations, K = ", s.y.K)

for k in (0, 5, 20)
    local t, s
    t = @elapsed s = solve_steady_state_joint(model; discretization, linsolve=:gmres, agent_steps=k, verbose=false)
    println("Newton-Krylov, agent_steps=$k: ", round(t, digits=3), " s, ", s.iterations,
        " iterations, GMRES: ", s.gmres_iterations, ", K = ", s.y.K)
end

(; r, w) = projection(model, sol.y, sol.z)
println("\nr = $r (target 0.01), K = $(sol.y.K) (target $K_ss)")
