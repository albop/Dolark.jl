# Sequence-space jacobian: checked against a direct solve of the stacked
# linear system E1-E4 over T periods (same truncation), on random systems.

using Random

function _ssj_random_linearization(rng; N=30, n_x=2, coupled=false)
    Nx = N * n_x
    n_p = 2; n_y = 2; n_z = 1
    # Π row-stochastic, 3 destinations per row
    Is = Int[]; Js = Int[]; Vs = Float64[]
    for n in 1:N
        d = rand(rng, 1:N, 3); w = rand(rng, 3); w ./= sum(w)
        append!(Is, fill(n, 3)); append!(Js, d); append!(Vs, w)
    end
    Π = sparse(Is, Js, Vs, N, N)
    # Fx block diagonal (or not, to test the fallback), FX small
    blocks = [3I + 0.3 * randn(rng, n_x, n_x) for _ in 1:N]
    Fx = blockdiag(sparse.(blocks)...)
    coupled && (Fx[1, Nx] = 0.2; Fx[Nx, 2] = -0.1)
    FX = sprandn(rng, Nx, Nx, 4 / Nx) * 0.3
    Fp = randn(rng, Nx, n_p); FP = randn(rng, Nx, n_p)
    # lotteries conserve mass: columns of Πx, Πp, ΠP sum to zero
    Ix = Int[]; Jx = Int[]; Vx = Float64[]
    for j in 1:Nx
        v = randn(rng)
        append!(Ix, rand(rng, 1:N, 2)); append!(Jx, [j, j]); append!(Vx, [v, -v])
    end
    Πx = sparse(Ix, Jx, Vx, N, Nx)
    center(A) = A .- sum(A; dims=1) ./ size(A, 1)
    Πp = center(randn(rng, N, n_p)); ΠP = center(randn(rng, N, n_p))
    # equation 1: K_{t+1} = ∫ a dμ_t (predetermined K), equation 2: L_t = ... (not)
    Aμ = [rand(rng, N)'; 0.1 * randn(rng, N)']
    Ax = 0.1 * randn(rng, 2, Nx); Ap = 0.1 * randn(rng, 2, n_p)
    Ay = [0.0 0.1; 0.2 -1.0]; AY = [-1.0 0.0; 0.1 0.2]
    Az = randn(rng, 2, n_z); AZ = randn(rng, 2, n_z)
    p_y = 0.3 * randn(rng, n_p, n_y); p_z = randn(rng, n_p, n_z)
    return Dolark.Linearization(nothing, nothing, nothing, (:r, :w), (:K, :L), (:z,),
        Fx, FX, Fp, FP, Π, Πx, Πp, ΠP, Aμ, Ax, Ap, Ay, AY, Az, AZ, p_y, p_z,
        fill(0.8, 1, 1), fill(0.01, 1, 1), [true, false])
end

# stacked system; unknowns dx_0..dx_{T-1}, dμ_1..dμ_{T-1} (and the unknown
# aggregates, ordered as in `ge_jacobians`); time-major price/exogenous paths.
function _ssj_stacked(lin, T; revaluation=false)
    N = size(lin.Π, 1); Nx = size(lin.Fx, 1)
    n_p = size(lin.Fp, 2)
    It = sparse(1.0I, T, T)
    S = spdiagm(T, T, 1 => ones(T - 1))                    # (S v)_t = v_{t+1}, v_T = 0
    Sel = sparse(1.0I, T - 1, T)                            # dates 0..T-2 of 0..T-1
    Lag = spdiagm(T - 1, T - 1, -1 => ones(T - 2))
    Mμ = [spzeros(1, T - 1); sparse(1.0I, T - 1, T - 1)]    # dμ_0 = 0
    P0 = kron(It, sparse(1.0I, n_p, n_p)); P1 = kron(S, sparse(1.0I, n_p, n_p))
    E1x = kron(It, lin.Fx) + kron(S, lin.FX)
    E1p = kron(It, sparse(lin.Fp)) * P0 + kron(It, sparse(lin.FP)) * P1
    E2x = -kron(Sel, lin.Πx)
    E2μ = kron(sparse(1.0I, T - 1, T - 1), sparse(1.0I, N, N)) - kron(Lag, sparse(lin.Π'))
    E2p = -kron(Sel, sparse(lin.Πp)) * P0 - kron(Sel, sparse(lin.ΠP)) * P1
    A_hh = [E1x spzeros(T * Nx, (T - 1) * N); E2x E2μ]
    B_hh = [E1p; E2p]
    Ox = kron(It, sparse(lin.Ax)); Oμ = kron(Mμ, sparse(lin.Aμ))
    # dμ_0 = ΠP dp_0 with revaluation: enters E2 at t = 0 and the outcome at t = 0
    E00 = sparse([1], [1], [1.0], T, T)
    Op0 = revaluation ? kron(E00, sparse(lin.Aμ * lin.ΠP)) * P0 : spzeros(size(lin.Aμ, 1) * T, n_p * T)
    if revaluation
        B_hh = B_hh - [spzeros(T * Nx, n_p * T); kron(E00[1:(T - 1), :], sparse(lin.Π' * lin.ΠP)) * P0]
    end
    return (; A_hh, B_hh, Ox, Oμ, Op0, It, S)
end

# household outcomes Aμ dμ_t + Ax dx_t for a given (time-major) price path
function _ssj_bruteforce_hh(lin, T, dP; revaluation=false)
    st = _ssj_stacked(lin, T; revaluation)
    u = st.A_hh \ Vector(-(st.B_hh * dP))
    Nx = size(lin.Fx, 1)
    O = [st.Ox st.Oμ] * u + st.Op0 * dP
    return reshape(O, size(lin.Aμ, 1), T)'      # T × n_A
end

# general equilibrium: dy (dates 0..T-1, T × n_y) given the exogenous path z (T × n_z)
function _ssj_bruteforce_ge(lin, T, z; revaluation=false)
    st = _ssj_stacked(lin, T; revaluation)
    n_y = size(lin.Ay, 2); n_z = size(lin.Az, 2); n_A = size(lin.Aμ, 1)
    Nx = size(lin.Fx, 1); N = size(lin.Π, 1)
    # dy at dates 0..T (time-major) from the unknowns
    Is = Int[]; Js = Int[]
    for i in 1:n_y, c in 1:T
        d = c - 1 + (lin.predetermined[i] ? 1 : 0)
        push!(Is, d * n_y + i); push!(Js, (i - 1) * T + c)
    end
    Yf = sparse(Is, Js, ones(length(Is)), (T + 1) * n_y, n_y * T)
    Y0 = Yf[1:(T * n_y), :]; Y1 = Yf[(n_y + 1):end, :]
    Zv = vec(Matrix(z'))
    dP_U = kron(st.It, sparse(lin.p_y)) * Y0               # prices: dp = p_y dy + p_z dz (dates 0..T-1)
    dP_Z = kron(st.It, sparse(lin.p_z)) * Zv
    E3_U = kron(st.It, sparse(lin.Ap)) * dP_U + kron(st.It, sparse(lin.Ay)) * Y0 + kron(st.It, sparse(lin.AY)) * Y1 + st.Op0 * dP_U
    E3_Z = kron(st.It, sparse(lin.Ap)) * dP_Z + kron(st.It, sparse(lin.Az)) * Zv + kron(st.S, sparse(lin.AZ)) * Zv + st.Op0 * dP_Z
    A = [st.A_hh st.B_hh*dP_U; st.Ox st.Oμ E3_U]
    b = -[st.B_hh * dP_Z; E3_Z]
    u = A \ Vector(b)
    U = u[(end - n_y * T + 1):end]
    return Matrix(reshape(Y0 * U, n_y, T)')
end

@testset "SSJ" begin
    rng = MersenneTwister(1)
    T = 25
    for (n_x, coupled, revaluation) in ((2, false, true), (1, false, true), (1, true, true), (2, false, false))
        lin = _ssj_random_linearization(rng; N=30, n_x, coupled)
        n_p = 2; n_A = 2
        @test (Dolark._block_diagonal_inverse(lin.Fx, n_x) === nothing) == coupled
        J = household_jacobians(lin, T; revaluation)
        @test size(J) == (n_A, n_p) && size(J[1, 1]) == (T, T)
        # (a) household jacobians, column by column
        err = 0.0; scale = 0.0
        for l in 1:n_p, s in (0, 1, 2, 7, T - 1)
            dP = zeros(T * n_p); dP[s * n_p + l] = 1.0
            O = _ssj_bruteforce_hh(lin, T, dP; revaluation)
            for a in 1:n_A
                err = max(err, maximum(abs, J[a, l][:, s + 1] - O[:, a]))
                scale = max(scale, maximum(abs, O[:, a]))
            end
        end
        @test err < 1e-10 * scale
        # (b) general equilibrium impulse responses
        shock = [0.7]
        r = irf_ssj(lin; T, shock, revaluation)
        z = [0.8^t * shock[1] for t in 0:(T - 1), _ in 1:1]
        y = _ssj_bruteforce_ge(lin, T, z; revaluation)
        @test r.z ≈ z
        @test maximum(abs, r.y - y) < 1e-10 * maximum(abs, y)
        @test r.y[1, 1] == 0                     # predetermined
        @test abs(r.y[1, 2]) > 0
        @test r.p ≈ r.y * lin.p_y' + r.z * lin.p_z'
        # G is reused
        g = ge_jacobians(lin, T; J)
        @test g.H_U * g.G ≈ -g.H_Z
        @test irf_ssj(lin; T, shock, G=g.G).y ≈ r.y
    end

    # Aiyagari model, when the linearization is available
    if isdefined(Dolark, :linearize)
        d = Dict(:endo => Dict(:n => 100))
        ss = solve_steady_state(model; discretization=d, verbose=false)
        lin = Dolark.linearize(model, ss)
        T = 300
        r = irf_ssj(lin; T, shock=[0.01], revaluation=false)   # dμ_0 = 0
        rv = irf_ssj(lin; T, shock=[0.01])                     # dμ_0 = ΠP dp_0 (default)
        @test r.y[1, 1] == 0 && rv.y[1, 1] == 0
        @test rv.y[2, 1] > 0                     # capital rises after a TFP shock
        @test abs(rv.y[end, 1]) < 1e-2 * maximum(abs, rv.y[:, 1])
        @test r.p[1, :] ≈ rv.p[1, :]             # date-0 prices: z only
        if isdefined(Dolark, :irf_reiter)
            sol = Dolark.solve_reiter(lin)
            rr = Dolark.irf_reiter(sol, lin; T, shock=[0.01], dμ0=zeros(size(lin.Π, 1)))
            rrv = Dolark.irf_reiter(sol, lin; T, shock=[0.01], dμ0=lin.ΠP * (lin.p_z * [0.01]))
            @test maximum(abs, rr.y[1:150, :] - r.y[1:150, :]) < 1e-6 * maximum(abs, r.y)
            @test maximum(abs, rrv.y[1:150, :] - rv.y[1:150, :]) < 1e-6 * maximum(abs, rv.y)
        end
    end
end
