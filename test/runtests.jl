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

@testset "Derivatives" begin
    D = Dolark
    d = Dict(:endo => Dict(:n => 50))
    y = (; K=15.0)
    res = excess(model, y; discretization=d)
    (; dmodel, sol, x, μ) = res
    φ = sol.dr
    N = length(x)
    names = D.price_names(model)
    p0 = D._current_prices(dmodel, names)
    pv = D._as_vector(p0)
    h = 1e-6
    reprice(u) = Dolo.discretize(D.set_exogenous(model.agent, NamedTuple{names}(Tuple(u))), d)
    e(l) = SVector(ntuple(i -> i == l ? h : 0.0, length(pv)))

    # pointwise residual is the same as Dolo's
    @test all(Dolo.F(dmodel, x, φ).data .≈ [D.household_residual(dmodel, n, x[n], φ, p0) for n in 1:N])
    @test transition_matrix(dmodel, x) == D.distribution_derivatives(dmodel, x, μ, names).Π

    # F_p
    (; J, L, F_p) = D.household_derivatives(dmodel, x, φ, names)
    for l in 1:length(pv)
        fd = (Dolo.F(reprice(pv + e(l)), x, φ).data - Dolo.F(reprice(pv - e(l)), x, φ).data) ./ (2h)
        @test maximum(maximum(abs, F_p[n][:, l] - fd[n]) for n in 1:N) < 1e-7
    end

    # household linear solve
    D.precondition!(J, L)
    b = Dolo.GArray(dmodel.grid, [SVector(sin(n)) for n in 1:N])
    dx = D.solve_household(J, L, b)
    L0 = Dolo.dF_2(dmodel, x, φ)
    J0 = Dolo.dF_1(dmodel, x, φ)
    Ax = Dolo.GArray(dmodel.grid, (J0.data .* dx.data)) 
    @test maximum(abs, D._flatten(Ax) + L0 * D._flatten(dx) - D._flatten(b)) < 1e-10

    # Π_x and Π_p
    (; Π_x, Π_p) = D.distribution_derivatives(dmodel, x, μ, names)
    for c in 1:7:N
        xp = deepcopy(x); xm = deepcopy(x)
        xp.data[c] += SVector(h); xm.data[c] -= SVector(h)
        fd = (transition_matrix(dmodel, xp)' * μ - transition_matrix(dmodel, xm)' * μ) / (2h)
        @test maximum(abs, Π_x[:, c] - fd) < 1e-8
    end
    for l in 1:length(pv)
        fd = (transition_matrix(reprice(pv + e(l)), x)' * μ - transition_matrix(reprice(pv - e(l)), x)' * μ) / (2h)
        @test maximum(abs, Π_p[:, l] - fd) < 1e-8
    end

    # total derivative
    dA = excess_jacobian(model, res)
    hK = 1e-3
    fd = (excess(model, (; K=y.K + hK); discretization=d, x0=x).A -
          excess(model, (; K=y.K - hK); discretization=d, x0=x).A) / (2hK)
    @test isapprox(dA[1], fd[1]; rtol=1e-4)

    # steady state with the finite difference jacobian gives the same result
    s1 = solve_steady_state(model; discretization=d, verbose=false)
    s2 = solve_steady_state(model; discretization=d, verbose=false, jacobian=:fd)
    @test isapprox(s1.y.K, s2.y.K; rtol=1e-6)
end

@testset "Newton-Krylov" begin
    D = Dolark
    d = Dict(:endo => Dict(:n => 50))
    z = (; z=0.0)
    res = excess(model, (; K=15.0); discretization=d)
    pt = D.JointPoint(model, (; K=15.0), z, res.x, res.μ, d)
    Jv!, Pinv!, n = D.joint_linearization(model, pt, z)
    u0 = D._pack(pt.x, pt.μ, pt.y)
    f(u) = (p = D._unpack(pt, u); D.joint_residual(model, D.JointPoint(model, p[3], z, p[1], p[2], d), z))
    v = [sin(i) for i in 1:n]
    h = 1e-6
    fd = (f(u0 + h * v) - f(u0 - h * v)) / (2h)
    @test maximum(abs, Jv!(zeros(n), v) - fd) < 1e-8 * maximum(abs, fd)

    s1 = solve_steady_state(model; discretization=d, verbose=false)
    for k in (0, 5)
        s2 = solve_steady_state_joint(model; discretization=d, agent_steps=k, verbose=false)
        @test maximum(abs, s2.A) < 1e-8
        @test isapprox(s1.y.K, s2.y.K; rtol=1e-6)
        @test sum(s2.μ) ≈ 1
    end
end
