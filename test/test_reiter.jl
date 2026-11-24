# Linearization around the steady state and Reiter's method.
# (included from runtests.jl, where `model` is the Aiyagari model)

module _TestReiterHelpers

using Dolark, Dolo, StaticArrays
const D = Dolark

# independent implementations of the nonlinear blocks (n_x = 1)

# agent's residuals F(x_t, x_{t+1}; p_t, p_{t+1}) on the grid
function agent_residual(lin, x::AbstractVector, xf::AbstractVector, p, pf)
    dm = lin.dmodel
    names = lin.price_names
    n_m = size(dm.grid)[1]
    nodes = D._nodes(D._grid1d(dm.grid.grids[2]))
    P = dm.dproc.P; Q = dm.dproc.Q
    K = size(P, 2)
    pn = NamedTuple{names}(Tuple(p)); pfn = NamedTuple{names}(Tuple(pf))
    map(1:length(x)) do n
        i, s = D._state(dm, n, pn)
        x_n = SVector(x[n])
        r = zero(SVector{1,Float64})
        for j in 1:K
            M = D._exo_values(D._exo_names(dm), Q[j], pfn)
            S_endo = SVector(Dolo.transition(dm.model, s.val, x_n, M)...)
            k = D._walk(nodes, S_endo[1], 1)
            λ = (S_endo[1] - nodes[k]) / (nodes[k+1] - nodes[k])
            X = SVector((1 - λ) * xf[D._linear_index(n_m, j, k)] + λ * xf[D._linear_index(n_m, j, k + 1)])
            S = Dolo.QP((j, S_endo), SVector(M..., S_endo...))
            r += P[i, j] * Dolo.arbitrage(dm, s, x_n, S, X)
        end
        Dolo.complementarities(dm.model, s, x_n, r)[1]
    end
end

# Π(x, p, pf)'μ (lotteries clamped at the bounds of the grid)
function forward(lin, μ, x::AbstractVector, p, pf)
    dm = lin.dmodel
    names = lin.price_names
    n_m = size(dm.grid)[1]
    g = D._grid1d(dm.grid.grids[2])
    P = dm.dproc.P; Q = dm.dproc.Q
    pn = NamedTuple{names}(Tuple(p)); pfn = NamedTuple{names}(Tuple(pf))
    out = zeros(length(μ))
    for n in eachindex(μ)
        i, s = D._state(dm, n, pn)
        for j in 1:size(P, 2)
            M = D._exo_values(D._exo_names(dm), Q[j], pfn)
            S = Dolo.transition(dm.model, s.val, SVector(x[n]), M)[1]
            k, λ = D._lottery(g, S)
            out[D._linear_index(n_m, j, k)] += μ[n] * P[i, j] * (1 - λ)
            out[D._linear_index(n_m, j, k + 1)] += μ[n] * P[i, j] * λ
        end
    end
    out
end

# Σ_n μ[n] equilibrium(s_n(p), x_n, y, z, Y, Z)
function equilibrium_sum(lin, μ, x::AbstractVector, p, y, z, Y, Z)
    dm = lin.dmodel
    m = lin.model
    yn(v) = NamedTuple{lin.aggregate_names}(Tuple(v))
    zn(v) = NamedTuple{lin.exogenous_names}(Tuple(v))
    pn = NamedTuple{lin.price_names}(Tuple(p))
    sum(μ[n] * Dolark.equilibrium(m, D._state(dm, n, pn)[2].val, SVector(x[n]), yn(y), zn(z), yn(Y), zn(Z))
        for n in eachindex(μ))
end

end # module

@testset "Linearization" begin
    H = _TestReiterHelpers
    D = Dolark
    d = Dict(:endo => Dict(:n => 100))
    ss = solve_steady_state(model; discretization=d, verbose=false)
    lin = linearize(model, ss)
    @test lin isa Linearization
    N = length(ss.x); Nx = N
    names = D.price_names(model, ss.y, ss.z)
    @test lin.price_names == names
    @test lin.aggregate_names == (:K,)
    @test lin.exogenous_names == (:z,)
    @test size(lin.Fx) == (Nx, Nx) && size(lin.FX) == (Nx, Nx)
    @test size(lin.Fp) == (Nx, 2) && size(lin.FP) == (Nx, 2)
    @test size(lin.Π) == (N, N) && size(lin.Πx) == (N, Nx)
    @test size(lin.Aμ) == (1, N) && size(lin.Ax) == (1, Nx)

    # consistency with the steady-state derivatives
    pass = agent_pass(ss.dmodel, ss.x, names)
    @test maximum(abs, lin.Fx + lin.FX - D.jacobian_matrix(pass)) < 1e-12
    @test maximum(abs, lin.Fp + lin.FP - D._stack_blocks(pass.Fp)) < 1e-12
    dd = D.distribution_derivatives(pass, ss.μ)
    @test maximum(abs, lin.Π - dd.Π) < 1e-14
    @test maximum(abs, lin.Πp + lin.ΠP - dd.Π_p) < 1e-12
    @test maximum(abs, lin.Πx - dd.Π_x) < 1e-14
    ad = D.aggregate_derivatives(model, ss.dmodel, ss.μ, ss.x, ss.y, ss.z, names)
    @test lin.Aμ ≈ ad.A_μ
    @test lin.Ax ≈ ad.A_x
    @test lin.Ay + lin.AY ≈ ad.A_y
    @test lin.Ap ≈ ad.A_p
    @test lin.Π' * ss.μ ≈ ss.μ
    # lotteries conserve mass
    @test maximum(abs, sum(lin.Πx; dims=1)) < 1e-12
    @test maximum(abs, sum(lin.Πp; dims=1)) < 1e-12
    @test maximum(abs, sum(lin.ΠP; dims=1)) < 1e-12
    # F_x is block diagonal
    @test all(((r, c),) -> r == c, zip(findnz(lin.Fx)[1:2]...))

    # finite differences of the nonlinear blocks
    x0 = D._flatten(ss.x)
    pv = collect(D._as_vector(D._current_prices(ss.dmodel, names)))
    h = 1e-6
    e(l, n) = [i == l ? 1.0 : 0.0 for i in 1:n]
    v = [sin(3i) for i in 1:Nx] .* 1e-1
    F0 = H.agent_residual(lin, x0, x0, pv, pv)
    @test maximum(abs, F0) < 1e-9
    rel(a, b) = maximum(abs, a - b) / max(maximum(abs, b), 1e-12)
    fd = (H.agent_residual(lin, x0 + h * v, x0, pv, pv) - H.agent_residual(lin, x0 - h * v, x0, pv, pv)) / 2h
    @test rel(lin.Fx * v, fd) < 1e-6
    fd = (H.agent_residual(lin, x0, x0 + h * v, pv, pv) - H.agent_residual(lin, x0, x0 - h * v, pv, pv)) / 2h
    @test rel(lin.FX * v, fd) < 1e-6
    for l in 1:2
        hp = h * e(l, 2)
        fd = (H.agent_residual(lin, x0, x0, pv + hp, pv) - H.agent_residual(lin, x0, x0, pv - hp, pv)) / 2h
        @test maximum(abs, lin.Fp[:, l] - fd) < 1e-6 * max(1, maximum(abs, fd))
        fd = (H.agent_residual(lin, x0, x0, pv, pv + hp) - H.agent_residual(lin, x0, x0, pv, pv - hp)) / 2h
        @test rel(lin.FP[:, l], fd) < 1e-6
    end

    μ = ss.μ
    @test H.forward(lin, μ, x0, pv, pv) ≈ lin.Π' * μ
    fd = (H.forward(lin, μ, x0 + h * v, pv, pv) - H.forward(lin, μ, x0 - h * v, pv, pv)) / 2h
    @test rel(lin.Πx * v, fd) < 1e-6
    for l in 1:2
        hp = h * e(l, 2)
        fd = (H.forward(lin, μ, x0, pv + hp, pv) - H.forward(lin, μ, x0, pv - hp, pv)) / 2h
        @test maximum(abs, lin.Πp[:, l] - fd) < 1e-8
        fd = (H.forward(lin, μ, x0, pv, pv + hp) - H.forward(lin, μ, x0, pv, pv - hp)) / 2h
        @test rel(lin.ΠP[:, l], fd) < 1e-6
    end

    yv = [ss.y.K]; zv = [ss.z.z]
    A(; μ=μ, x=x0, p=pv, y=yv, z=zv, Y=yv, Z=zv) = collect(H.equilibrium_sum(lin, μ, x, p, y, z, Y, Z))
    @test maximum(abs, A()) < 1e-6
    vμ = [cos(i) for i in 1:N]
    @test lin.Aμ * vμ ≈ (A(μ=μ + h * vμ) - A(μ=μ - h * vμ)) / 2h
    @test lin.Ax * v ≈ (A(x=x0 + h * v) - A(x=x0 - h * v)) / 2h atol=1e-8
    for l in 1:2
        hp = h * e(l, 2)
        @test lin.Ap[:, l] ≈ (A(p=pv + hp) - A(p=pv - hp)) / 2h atol=1e-8
    end
    @test vec(lin.Ay) ≈ (A(y=yv .+ h) - A(y=yv .- h)) / 2h atol=1e-8
    @test vec(lin.AY) ≈ (A(Y=yv .+ h) - A(Y=yv .- h)) / 2h rtol=1e-6
    @test vec(lin.Az) ≈ (A(z=zv .+ h) - A(z=zv .- h)) / 2h atol=1e-8
    @test vec(lin.AZ) ≈ (A(Z=zv .+ h) - A(Z=zv .- h)) / 2h atol=1e-8

    pr(y, z) = collect(values(projection(model, (; K=y), (; z=z))))
    @test lin.p_y[:, 1] ≈ (pr(ss.y.K + h, ss.z.z) - pr(ss.y.K - h, ss.z.z)) / 2h rtol=1e-6
    @test lin.p_z[:, 1] ≈ (pr(ss.y.K, ss.z.z + h) - pr(ss.y.K, ss.z.z - h)) / 2h rtol=1e-6
    @test lin.ρ == fill(0.9, 1, 1)
    @test lin.Σ ≈ fill(0.01^2, 1, 1)

    # K is predetermined (automatic detection), and overrides
    @test lin.predetermined == [true]
    @test linearize(model, ss; predetermined=(:K,)).predetermined == [true]
    @test linearize(model, ss; predetermined=[false]).predetermined == [false]
    @test_throws ErrorException linearize(model, ss; predetermined=(:Q,))
end

@testset "Reiter" begin
    D = Dolark
    d = Dict(:endo => Dict(:n => 50))
    ss = solve_steady_state(model; discretization=d, verbose=false)
    lin = linearize(model, ss)
    N = size(lin.Π, 1)
    sol = solve_reiter(lin; reduction=:none)
    @test sol isa ReiterSolution && sol.reduction === :none
    n = N + N - 1 + 1 + 1                            # (dx, dμ̃, dK, dz)
    @test size(sol.X) == (n, n) && size(sol.Y) == (n, 1)
    @test sol.shifted == [true]                      # K_t = ∫ a_{t-1} dμ_{t-1}
    # the solution solves the quadratic equation, Blanchard-Kahn holds
    (; A, B, C) = sol
    @test maximum(abs, A * sol.X^2 + B * sol.X + C) < 1e-10
    evs = sol.cr.eigenvalues
    @test length(evs) == 2n && count(<=(1 + 1e-6), evs) == n
    @test sol.cr.stable_radius < 1 < sol.cr.unstable_min

    # generic cyclic reduction against a QZ decomposition of the companion pencil
    X_qz = let
        Z = zeros(n, n); Id = Matrix(1.0I, n, n)
        F = schur([Z Id; -C -B], [Id Z; Z A])
        λ = F.α ./ F.β
        ordschur!(F, abs.(λ) .<= 1 + 1e-6)
        F.Z[n+1:end, 1:n] / F.Z[1:n, 1:n]
    end
    @test maximum(abs, sol.X - X_qz) < 1e-8 * maximum(abs, sol.X)
    @test solve_cr(A, B, C; tau=0.31).X ≈ sol.X     # Möbius shift (small systems)

    T = 200
    shock = D._default_shock(lin)
    irf = irf_reiter(sol, lin; T=T)
    @test irf isa IRF
    @test size(irf.y) == (T, 1) && size(irf.p) == (T, 2) && size(irf.z) == (T, 1)
    @test irf.y[1, 1] == 0.0                         # K is predetermined
    @test irf.z[:, 1] ≈ shock[1] .* 0.9 .^ (0:T-1)
    @test irf.p ≈ irf.y * lin.p_y' + irf.z * lin.p_z'
    @test irf_reiter(lin; T=T, reduction=:none).y ≈ irf.y

    # same as the brute-force perfect-foresight solution (stacked over T periods;
    # the truncation error decays slowly: largest stable |λ| ≈ 0.996)
    T = 300
    irf = irf_reiter(sol, lin; T=T)
    st = D._irf_stacked(lin; T=T)
    @test maximum(abs, irf.y[1:50, :] - st.irf.y[1:50, :]) < 1e-8 * maximum(abs, irf.y)
    @test maximum(abs, irf.p[1:50, :] - st.irf.p[1:50, :]) < 1e-8 * maximum(abs, irf.p)
    V, L = D._reiter_paths(sol, lin, T, shock)
    @test maximum(abs, V[L.jx, 1:50] - st.dx[:, 1:50]) < 1e-7 * maximum(abs, st.dx)
    dμ = [V[L.jμ, :]; -sum(V[L.jμ, :]; dims=1)]
    @test maximum(abs, dμ[:, 1:50] - st.dμ[:, 1:50]) < 1e-7 * maximum(abs, st.dμ)

    # initial condition dμ_0 = ΠP dp_0 (prices depend only on K, predetermined, and z)
    dμ0 = lin.ΠP * (lin.p_z * shock)
    @test abs(sum(dμ0)) < 1e-14
    @test dμ[:, 1] ≈ dμ0
    @test irf_reiter(sol, lin; T=T, dμ0=dμ0).y ≈ irf.y
    # override: date-0 states do not respond to date-0 prices
    irf_old = irf_reiter(sol, lin; T=T, dμ0=zeros(N))
    st_old = D._irf_stacked(lin; T=T, dμ0=zeros(N))
    @test maximum(abs, irf_old.y[1:50, :] - st_old.irf.y[1:50, :]) < 1e-8 * maximum(abs, irf_old.y)
    @test irf_old.y[2, 1] < 0 < irf.y[2, 1]      # K_1 falls without the date-0 income effect

    # Blanchard-Kahn fails if K is declared a jump
    @test_throws ErrorException solve_reiter(linearize(model, ss; predetermined=[false]))

    # minimal state (default when all aggregates are predetermined): exact
    sm = solve_reiter(lin)
    @test sm.reduction === :minimal && size(sm.X) == (n, n)
    @test sm.cr.stable_radius ≈ sol.cr.stable_radius
    @test maximum(abs, irf_reiter(sm, lin; T=T).y - irf.y) < 1e-10 * maximum(abs, irf.y)
    @test irf_reiter(sm, lin; T=T).y[1, 1] == 0.0
    @test maximum(abs, irf_reiter(sm, lin; T=T, dμ0=zeros(N)).y - irf_old.y) < 1e-10 * maximum(abs, irf_old.y)
    @test_throws ErrorException solve_reiter(linearize(model, ss; predetermined=[false]); reduction=:minimal)

    # two-sided Krylov reduction: adaptive k, and fixed k
    sk = solve_reiter(lin; reduction=:krylov)
    @test sk.reduction === :krylov && size(sk.X, 1) < n
    @test size(sk.U, 2) + size(sk.V, 2) + 2 == size(sk.X, 1)
    @test sk.U' * sk.U ≈ I && sk.V' * sk.V ≈ I
    irf_k = irf_reiter(sk, lin; T=T)
    @test maximum(abs, irf_k.y - irf.y) < 1e-8 * maximum(abs, irf.y)
    @test irf_k.y[1, 1] == 0.0
    @test maximum(abs, irf_reiter(sk, lin; T=T, dμ0=zeros(N)).y - irf_old.y) < 1e-8 * maximum(abs, irf_old.y)
    s10 = solve_reiter(lin; reduction=:krylov, k=10)
    @test size(s10.U, 2) == size(s10.V, 2) == 10
    @test maximum(abs, irf_reiter(s10, lin; T=T).y - irf.y) < 0.1 * maximum(abs, irf.y)
    @test_throws ErrorException solve_reiter(lin; reduction=:qz)
end
