using Test
using Dolark
using Dolo
using StaticArrays

model = include(joinpath(@__DIR__, "..", "examples", "ayiagari.jl"))

@testset "HModel representation" begin
    @test isbits(model)
    @test Dolo.name(model) == :ayiagari
    @test Dolark.get_aggregate(model) == (:K,)
    @test Dolark.get_exogenous(model) == (:z,)

    y = (; K=20.0)
    z = (; z=0.0)
    p = projection(model, y, z)
    @test keys(p) == (:r, :w)

    # setting prices preserves the type of the agent
    agent = agent_model(model, y, z)
    @test typeof(agent) == typeof(model.agent)
    @test agent.calibration.r == p.r
    @test agent.calibration.w == p.w
    @test all(q -> q[1] == p.w && q[2] == p.r, agent.exogenous.Q)
    @test all(q -> q[1] == p.w && q[2] == p.r, agent.states.exo.points)
    # idiosyncratic values are untouched
    @test [q[3] for q in agent.exogenous.Q] == [q[3] for q in model.agent.exogenous.Q]
end

@testset "Distribution" begin
    d = Dict(:endo => Dict(:n => 50))
    res = excess(model, (; K=20.0); discretization=d)
    Π = transition_matrix(res.dmodel, res.x)
    @test size(Π) == (150, 150)
    @test all(≈(1.0), sum(Π; dims=2))
    @test all(>=(0), Π)
    @test sum(res.μ) ≈ 1.0
    @test all(>=(-1e-12), res.μ)
    @test res.μ' * Π ≈ res.μ'
    # enumeration order of the grid matches the linear index used by the distribution
    n_m = size(res.dmodel.grid)[1]
    @test all(
        n == Dolark._linear_index(n_m, ss.loc...)
        for (n, ss) in enumerate(Dolo.enum(res.dmodel.grid))
    )
end

@testset "Steady state" begin
    d = Dict(:endo => Dict(:n => 100))
    sol = solve_steady_state(model; discretization=d, verbose=false)
    @test maximum(abs, sol.A) < 1e-6
    (; r) = projection(model, sol.y, sol.z)
    β = model.agent.calibration.β
    @test -model.calibration.δ < r < 1/β - 1
    # little mass at the upper bound of the grid
    @test sum(sol.μ[end-size(sol.dmodel.grid)[1]+1:end]) < 1e-6
end
