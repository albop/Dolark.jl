# Aiyagari steady state with the baseline calibration of HABenchmark
# (`../HABenchmark/models/aiyagari/calibration.toml`), for a timing comparison
# with the sequence-space jacobian toolkit (`results/aiyagari/RESULTS.md` there):
#
#     β=0.96, EIS=0.5, α=0.36, δ=0.08, ρ=0.9, σ_ε=0.2 (innovations), 7 Rouwenhorst
#     states (E[e]=1), 500 asset points on [0, 200], log-spaced with pivot 0.25.
#
# Differences with the toolkit:
# - the endogenous state is cash on hand y = (1+r)a + w e (log-spaced grid in y
#   with the same pivot), whose bounds are set at the equilibrium prices of the toolkit;
# - we solve for K (initial guess 6.0, i.e. r ≈ 3.5%), not for r by bisection.
#
#     julia --project=. examples/ayiagari_habenchmark.jl

using Dolark, Dolo

include("ayiagari.jl")

β = 0.96; γ = 2.0; α = 0.36; δ = 0.08; ρ = 0.9; σ = 0.2
n_e = 7; n_a = 500; a_max = 200.0

# bounds of the grid in cash on hand, at the equilibrium prices of the toolkit
r_ref = 0.025369
K_ref = (α / (r_ref + δ))^(1 / (1 - α))
w_ref = (1 - α) * K_ref^α

model = let
    m = ayiagari_model(; β, γ, ρ, σ, n_e, normalize_e=true, α, δ, K=6.0)
    e = [exp(q[3]) for q in m.agent.exogenous.Q]
    y_min = w_ref * minimum(e)                    # a = 0
    y_max = (1 + r_ref) * a_max + w_ref * maximum(e)  # a = a_max
    ayiagari_model(; β, γ, ρ, σ, n_e, normalize_e=true, y_min, y_max, α, δ, K=6.0)
end

discretization = Dict(:endo => Dict(:y => Dolo.LogGrid(n_a; pivot=0.25)))

t_first = @elapsed solve_steady_state(model; discretization, verbose=false)
solve_steady_state_joint(model; discretization, verbose=false)
println("first call (with compilation): ", round(t_first, digits=2), " s")

function timing(f; n=10)
    ts = [@elapsed(f()) for _ in 1:n]
    return (median=sort(ts)[n÷2+1], min=minimum(ts))
end
ms(t) = string(round(1e3 * t, digits=1), " ms")

t = timing(() -> solve_steady_state(model; discretization, verbose=false))
sol = solve_steady_state(model; discretization, verbose=false)
println("nested Newton (exact jacobian): ", ms(t.median), " (min ", ms(t.min), "), ",
    sol.iterations, " iterations, K = ", sol.y.K)

t = timing(() -> solve_steady_state_joint(model; discretization, verbose=false))
s = solve_steady_state_joint(model; discretization, verbose=false)
println("joint Newton (direct):          ", ms(t.median), " (min ", ms(t.min), "), ",
    s.iterations, " iterations, K = ", s.y.K)

# household block at fixed prices, cold start (as `hh_block` in the toolkit's report)
y = sol.y
t = timing(() -> excess(model, y; discretization))
println("agent + distribution at fixed prices (cold start): ", ms(t.median))

(; r, w) = projection(model, sol.y, sol.z)
println("\nr = $r, w = $w (toolkit: r = 0.025369, w = 1.277392, K = 6.819205)")
