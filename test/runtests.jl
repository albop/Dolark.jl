using Test
using Dolark
using Dolo
using StaticArrays
using LinearAlgebra
using SparseArrays

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
        s2 = solve_steady_state_joint(model; discretization=d, agent_steps=k, linsolve=:gmres, verbose=false)
        @test maximum(abs, s2.A) < 1e-8
        @test isapprox(s1.y.K, s2.y.K; rtol=1e-6)
        @test sum(s2.μ) ≈ 1
    end

    # exact Newton steps (block elimination) solve the linearized system
    R = D.joint_residual(model, pt, z)
    du = D.joint_newton_step(model, pt, z, R).du
    @test maximum(abs, Jv!(zeros(n), du) + R) < 1e-9 * maximum(abs, R)
    s3 = solve_steady_state_joint(model; discretization=d, linsolve=:direct, verbose=false)
    @test maximum(abs, s3.A) < 1e-8
    @test isapprox(s1.y.K, s3.y.K; rtol=1e-6)
    @test sum(s3.μ) ≈ 1
end

@testset "EGM cold start" begin
    D = Dolark
    d = Dict(:endo => Dict(:n => 100))
    dm = Dolo.discretize(agent_model(model, (; K=20.0)), d)
    xe = egm(dm)
    @test xe !== nothing
    @test D._isfinite(xe)
    s_egm = solve_agent(dm)                    # uses the cold start (EGM)
    s_ti = D._accelerated_ti(dm)               # without it
    @test s_egm.ε < 1e-10
    @test maximum(abs, D._flatten(s_egm.dr.values) - D._flatten(s_ti.dr.values)) < 1e-7
    # EGM is close to the solution on Dolo's grid (they differ near the
    # borrowing constraint, where the uniform grid is coarse)
    gap = sort(abs.(D._flatten(xe) - D._flatten(s_ti.dr.values)))
    @test gap[end÷2] < 1e-2
end

@testset "Generic EGM" begin
    D = Dolark
    d = Dict(:endo => Dict(:n => 100))
    dm = Dolo.discretize(agent_model(model, (; K=20.0)), d)
    r = egm(dm; details=true, tol=1e-10, a_grid=D._poststate_grid(poststates(dm.model), 300))
    @test r isa EGMResult
    @test r.converged
    @test all(issorted, eachcol(r.s_grid))
    # on the endogenous grid, the Euler equation holds where the constraint does
    # not bind: residuals of Dolo's arbitrage equations at the EGM solution
    φ = D._decision_rule(dm, r.x)
    res = D._flatten(Dolo.F(dm, r.x, φ))
    @test sort(abs.(res))[end÷2] < 1e-4
    # close to the solution on Dolo's grid (up to the interpolation error of
    # the latter, with a uniform step of 1 in cash on hand)
    s = solve_agent(dm)
    gap = abs.(D._flatten(r.x) - D._flatten(s.dr.values))
    @test sort(gap)[end÷2] < 2e-3
    # the generic loop over (m, M) gives the same fixed point as the shortcut
    # used when the half transition does not depend on m
    p = D.EGMProblem(dm.model)
    A = [SVector(a) for a in r.a_grid]
    Q = dm.dproc.Q; P = Matrix(dm.dproc.P)
    Z1 = fill(zero(SVector{1,Float64}), length(A), length(Q))
    H = similar(Z1)
    D._egm_expectations_indep!(p, Z1, H, r.s_grid, r.x_grid, A, Q, P)
    Z2 = similar(Z1)
    for i in eachindex(Q)
        D._egm_expectations!(p, Z2, i, i, r.s_grid, r.x_grid, A, Q, P)
    end
    @test maximum(maximum.(abs, Z1 - Z2)) < 1e-12
    # a fixed point: one more iteration does not change the endogenous grid
    hn = Val((:mr,))
    x1 = [D._direct_response(p, Q[i], A[l], D._nt(hn, Z2[l, i])) for l in eachindex(A), i in eachindex(Q)]
    @test maximum(maximum.(abs, x1 - r.x_grid)) < 1e-9
end

@testset "Non-uniform grid" begin
    D = Dolark
    d = Dict(:endo => Dict(:y => Dolo.LogGrid(60; pivot=0.25)))
    z = (; z=0.0)
    res = excess(model, (; K=15.0); discretization=d)
    @test res.dmodel.grid.grids[2] isa Dolo.CartesianGrid{1}
    Π = transition_matrix(res.dmodel, res.x)
    @test all(≈(1.0), sum(Π; dims=2))
    @test sum(res.μ) ≈ 1

    # exact jacobian of the full system
    pt = D.JointPoint(model, (; K=15.0), z, res.x, res.μ, d)
    Jv!, Pinv!, n = D.joint_linearization(model, pt, z)
    u0 = D._pack(pt.x, pt.μ, pt.y)
    f(u) = (p = D._unpack(pt, u); D.joint_residual(model, D.JointPoint(model, p[3], z, p[1], p[2], d), z))
    v = [cos(i) for i in 1:n]
    h = 1e-6
    fd = (f(u0 + h * v) - f(u0 - h * v)) / (2h)
    @test maximum(abs, Jv!(zeros(n), v) - fd) < 1e-7 * maximum(abs, fd)

    # dA/dK
    dA = excess_jacobian(model, res)
    hK = 1e-3
    fdK = (excess(model, (; K=15.0 + hK); discretization=d, x0=res.x).A -
           excess(model, (; K=15.0 - hK); discretization=d, x0=res.x).A) / (2hK)
    @test isapprox(dA[1], fdK[1]; rtol=1e-4)

    # all solvers agree
    s1 = solve_steady_state(model; discretization=d, verbose=false)
    s2 = solve_steady_state_joint(model; discretization=d, agent_steps=5, linsolve=:gmres, verbose=false)
    s3 = solve_steady_state_joint(model; discretization=d, verbose=false)
    s4 = solve_steady_state(model; discretization=d, jacobian=:fd, verbose=false)
    @test maximum(abs, s2.A) < 1e-8
    @test maximum(abs, s3.A) < 1e-8
    @test isapprox(s1.y.K, s2.y.K; rtol=1e-6)
    @test isapprox(s1.y.K, s3.y.K; rtol=1e-6)
    @test isapprox(s1.y.K, s4.y.K; rtol=1e-6)
end

@testset "Sparse agent jacobian" begin
    D = Dolark
    for d in (Dict(:endo => Dict(:n => 50)), Dict(:endo => Dict(:y => Dolo.LogGrid(60; pivot=0.25))))
        dm = Dolo.discretize(agent_model(model, (; K=15.0)), d)
        @test D._sparse_jacobian_available(dm)
        x = egm(dm)
        φ = D._decision_rule(dm, x)
        r, M = agent_jacobian(dm, x, φ)
        N = length(x)
        @test size(M) == (N, N)
        @test D._flatten(r) ≈ D._flatten(Dolo.F(dm, x, φ))
        # same as Dolo's operators: (F_x + F_X) v
        v = [sin(3i) for i in 1:N]
        J0 = Dolo.dF_1(dm, x, φ)
        L0 = Dolo.dF_2(dm, x, φ)
        Jv = D._flatten(Dolo.GArray(dm.grid, J0.data .* reinterpret(SVector{1,Float64}, v)))
        @test maximum(abs, M * v - (Jv + L0 * v)) < 1e-10 * maximum(abs, M * v)
        # and as finite differences of x ↦ F(x, φ(x))
        h = 1e-6
        Fx(u) = (xu = Dolo.GArray(dm.grid, copy(reinterpret(SVector{1,Float64}, u))); D._flatten(Dolo.F(dm, xu, D._decision_rule(dm, xu))))
        u0 = D._flatten(x)
        fd = (Fx(u0 + h * v) - Fx(u0 - h * v)) / (2h)
        @test maximum(abs, M * v - fd) < 1e-6 * maximum(abs, fd)

        # Newton with the sparse solver, from the cold start and warm-started
        # with a factorization at another point
        s = newton_sparse(dm, x)
        @test s.ε < 1e-10
        s_ti = D._accelerated_ti(dm)
        @test maximum(abs, D._flatten(s.dr.values) - D._flatten(s_ti.dr.values)) < 1e-7
        dm2 = Dolo.discretize(agent_model(model, (; K=15.5)), d)
        s2 = newton_sparse(dm2, s.dr.values; factorization=lu(agent_jacobian(dm, s.dr.values, s.dr)[2]))
        @test s2.ε < 1e-10
    end
end

@testset "Distribution solver" begin
    D = Dolark
    d = Dict(:endo => Dict(:y => Dolo.LogGrid(60; pivot=0.25)))
    res = excess(model, (; K=15.0); discretization=d)
    Π = res.Π
    N = size(Π, 1)
    # reference: dense mass condition
    A = sparse(I - Π')
    A = [A[1:end-1, :]; ones(1, N)]
    e = zeros(N); e[end] = 1.0
    μ_ref = A \ e
    Df = D._distribution_factorization(Π)
    @test Df.n0 > 0
    @test ergodic_distribution(Df) ≈ μ_ref
    @test res.μ ≈ μ_ref
    r = [cos(i) for i in 1:N]
    for m in (0.0, 0.3)
        dμ = D._solve_mass(Df, r, m)
        @test dμ ≈ A \ vcat(r[1:end-1], m)
    end
    # preconditioned by the factorization at a nearby point
    res2 = excess(model, (; K=15.01); discretization=d, x0=res.x)
    Df2 = D._distribution_factorization(res2.Π, Df)
    A2 = sparse(I - res2.Π')
    A2 = [A2[1:end-1, :]; ones(1, N)]
    @test ergodic_distribution(Df2) ≈ A2 \ e
    @test D._solve_mass(Df2, r, 0.0) ≈ A2 \ vcat(r[1:end-1], 0.0)
end

@testset "Fused agent pass" begin
    D = Dolark
    for d in (Dict(:endo => Dict(:n => 50)), Dict(:endo => Dict(:y => Dolo.LogGrid(60; pivot=0.25))))
        res = excess(model, (; K=15.0); discretization=d)
        (; dmodel, x, μ) = res
        φ = res.sol.dr
        names = D.price_names(model)
        p = agent_pass(dmodel, x, names)
        # residuals and transitions, as computed by Dolo / pointwise
        @test D._flatten(p) ≈ D._flatten(Dolo.F(dmodel, x, φ)) atol=1e-13
        @test agent_pass(dmodel, x; derivatives=false).r ≈ p.r
        Π0 = transition_matrix(dmodel, x)
        @test maximum(abs, D.transition_matrix(p) - Π0) < 1e-14
        @test D.forward(p, μ) ≈ Π0' * μ
        # derivatives, as computed separately
        F_p = D.price_derivatives(dmodel, x, φ, names)
        @test maximum(maximum(abs, a - b) for (a, b) in zip(p.Fp, F_p.data)) < 1e-12
        old = D.distribution_derivatives(dmodel, x, μ, names)
        new = D.distribution_derivatives(p, μ)
        @test maximum(abs, new.Π_x - old.Π_x) < 1e-12
        @test maximum(abs, new.Π_p - old.Π_p) < 1e-12
        M = D.jacobian_matrix(p)
        v = [sin(3i) for i in 1:length(x)]
        J0 = Dolo.dF_1(dmodel, x, φ)
        L0 = Dolo.dF_2(dmodel, x, φ)
        Jv = D._flatten(Dolo.GArray(dmodel.grid, J0.data .* reinterpret(SVector{1,Float64}, v)))
        @test maximum(abs, M * v - (Jv + L0 * v)) < 1e-10 * maximum(abs, M * v)
        # stationarity system
        n0 = res.Df.n0
        @test D._distribution_matrix(p, n0) ≈ D._distribution_matrix(sparse(Π0), n0)
        @test D._distribution_matrix(p, 0) ≈ D._distribution_matrix(sparse(Π0), 0)
    end
end
