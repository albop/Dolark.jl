# Steady state of the Aiyagari model.
#
#     julia --project=. examples/ayiagari_steady_state.jl

using Dolark

model = include("ayiagari.jl")

println(model)

# prices implied by the initial guess
y0 = (; K=model.calibration.K)
z0 = (; z=model.calibration.z)
@show projection(model, y0, z0)

discretization = Dict(:endo => Dict(:n => 200))

# aggregate savings minus capital, at the initial guess
res = excess(model, y0; discretization)
@show res.A

sol = solve_steady_state(model; discretization)

(; K) = sol.y
(; r, w) = projection(model, sol.y, sol.z)
println()
println("Steady state: K = $K, r = $r, w = $w")
println("1/β - 1 = $(1/model.agent.calibration.β - 1)")

# Same steady state, solving for (x, μ, K) jointly with Newton's method (exact
# steps, by block elimination with sparse factorizations)
sol_joint = solve_steady_state_joint(model; discretization)
println("Joint Newton: K = $(sol_joint.y.K)")

# ... or with Newton-Krylov, where GMRES is preconditioned by the agent's block
# (with 5 Neumann terms)
sol_nk = solve_steady_state_joint(model; discretization, linsolve=:gmres, agent_steps=5)
println("Newton-Krylov: K = $(sol_nk.y.K), GMRES iterations: $(sol_nk.gmres_iterations)")
