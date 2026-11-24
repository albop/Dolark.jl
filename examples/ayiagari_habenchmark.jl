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

# linearized dynamics (sequence-space jacobian), TFP shock dZ_0 = 0.01, ρ_Z = 0.9, T = 300
# (the toolkit's K_t is end-of-period capital, i.e. our K_{t+1})
T = 300
lin = linearize(model, sol)
t_lin = timing(() -> linearize(model, sol))
t_J = timing(() -> household_jacobians(lin, T))
J = household_jacobians(lin, T)
t_G = timing(() -> ge_jacobians(lin, T; J))
G = ge_jacobians(lin, T; J).G
t_irf = timing(() -> irf_ssj(lin; T, shock=[0.01], G))
irf = irf_ssj(lin; T, shock=[0.01], G)
println("\nlinearize: ", ms(t_lin.median), ", household jacobians: ", ms(t_J.median),
    " (toolkit: 166 ms), GE jacobian: ", ms(t_G.median), " (toolkit: 12.5 ms), IRF: ",
    round(1e6 * t_irf.median, digits=1), " µs")
println("peak dK = ", maximum(irf.y[:, 1]), " (toolkit: 0.057046)")

# Reiter's method with two-sided Krylov reduction (adaptive k), same IRF (the
# difference, ~2e-8, is the truncation of SSJ at T = 300: 1e-14 against T = 1200)
solve_reiter(lin; reduction=:krylov)
t_R = timing(() -> solve_reiter(lin; reduction=:krylov); n=3)
sR = solve_reiter(lin; reduction=:krylov)
irf_R = irf_reiter(sR, lin; T, shock=[0.01])
println("Reiter (Krylov, ", size(sR.X, 1), " variables): ", ms(t_R.median),
    ", max |IRF - SSJ| / max |IRF| = ", maximum(abs, irf_R.y - irf.y) / maximum(abs, irf.y))
