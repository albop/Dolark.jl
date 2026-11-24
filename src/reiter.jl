# Reiter's method: the linearized model (E1-E4, see `linearization_types.jl`)
# solved as a (large) linear rational expectations model in Dynare's form
#
#     A E_t v_{t+1} + B v_t + C v_{t-1} + D ε_t = 0
#
# whose solution v_t = X v_{t-1} + Y ε_t is computed by cyclic reduction
# (`solve_cr`), on dense coefficients. Three formulations (`reduction`):
#
# `:none`, v = (dx, dμ̃, dy, dz):
# - The distribution has mass one, so Σ dμ = 0: we keep its first N-1 entries,
#   dμ = [I; -1'] dμ̃, and the first N-1 rows of E2 (this removes the unit root
#   of Π').
# - E2 is written at date t: dμ_t = Π' dμ_{t-1} + Πx dx_{t-1} + Πp dp_{t-1} + ΠP dp_t.
#   The date-0 surprise in prices then moves the date-0 distribution,
#   dμ_0 = ΠP dp_0, as required by the convention of `linearization_types.jl`
#   (including when dp_0 depends on non-predetermined aggregates).
# - Predetermined aggregates are encoded by the timing: the equilibrium
#   conditions whose only leads are predetermined aggregates (e.g.
#   K_{t+1} = ∫ a_t dμ_t) are written one period earlier (K_t = ∫ a_{t-1} dμ_{t-1}),
#   so that these aggregates do not respond to date-t innovations.
#
# `:minimal` (exact, all aggregates predetermined), v = (dx, dν̃, dκ, dz): the
# end-of-period distribution ν_t = Π'μ_t + Πx x_t + Πp p_t (μ_{t+1} = ν_t + ΠP p_{t+1})
# and κ_t = y_{t+1} are the only states: fewer variables appear with a lag.
#
# `:krylov` (approximate), v = (ξ, m, dy, dz): two-sided Krylov (moment
# matching) reduction.
# - Distribution: only its components seen by the aggregates matter,
#   m = V'dμ with V an orthonormal basis of the Krylov space of Π from
#   the (centred) rows of Aμ (the observability space): m_t = H'm_{t-1} + V'(...),
#   H = V'ΠV. These are the "expectation vectors" of the fake-news algorithm.
# - Policy: only its responses to prices matter, dx = U ξ with U an
#   orthonormal basis of the Krylov space of M = -Fx⁻¹FX from -Fx⁻¹[Fp FP]
#   (the controllability space): ξ_t = G ξ_{t+1} + U'(-Fx⁻¹)(Fp dp_t + FP dp_{t+1}),
#   G = U'MU. These span the policy responses to future price shocks of SSJ.
# The dimension k of both bases increases until the impulse responses of the
# aggregates converge (`krylov_tol`); 100-300 is typical, independently of
# the size of the grid.
#
# Cost: dense O(n³) operations per iteration of cyclic reduction, about a
# dozen iterations (quadratic convergence), with n ≈ (n_x + 1) N (`:none`,
# `:minimal`) or n ≈ 2k (`:krylov`).

export ReiterSolution, solve_reiter, irf_reiter

"""
    ReiterSolution

Solution of the linearized model by Reiter's method: `v_t = X v_{t-1} + Y ε_t`,
with `v` given by `reduction`:
- `:none`: `v = (dx, dμ̃, dy, dz)` (`dμ̃`: the first N-1 entries of dμ);
- `:minimal`: `v = (dx, dν̃, dκ, dz)` with `ν_t` the end-of-period
  distribution (`dμ_{t+1} = dν_t + ΠP dp_{t+1}`) and `κ_t = y_{t+1}`;
- `:krylov`: `v = (ξ, m, dy, dz)`, `dx ≈ U ξ`, `m ≈ V'dμ`.
Also contains the solver's output `cr` (a `CRSolution`: eigenvalues or
spectral radii, iterations, residual), the coefficients `A`, `B`, `C`, `D`
(dense), the positions of the blocks in `v` (`layout`), and the bases `U`,
`V` (`:krylov` only).
"""
struct ReiterSolution
    X::Matrix{Float64}
    Y::Matrix{Float64}
    cr::CRSolution
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
    N::Int
    n_x::Int
    shifted::Vector{Bool}      # equilibrium conditions written one period earlier
    reduction::Symbol
    layout::NamedTuple         # jx, jμ, jy, jz, n
    U::Union{Nothing, Matrix{Float64}}
    V::Union{Nothing, Matrix{Float64}}
end

function Base.show(io::IO, s::ReiterSolution)
    print(io, "ReiterSolution (", s.reduction, "): ", size(s.X, 1), " variables")
    s.reduction === :krylov && print(io, " (k = ", size(s.U, 2), ", ", size(s.V, 2), ")")
    print(io, ", cyclic reduction in ", s.cr.iterations,
        " iterations; largest stable |λ| = ", round(s.cr.stable_radius; sigdigits=6),
        ", smallest unstable |λ| = ", round(s.cr.unstable_min; sigdigits=6))
end

# positions of blocks of sizes (kx, kμ, n_y, n_z) in v
function _layout(kx, kμ, n_y, n_z)
    jx = 1:kx
    jμ = kx .+ (1:kμ)
    jy = kx + kμ .+ (1:n_y)
    jz = kx + kμ + n_y .+ (1:n_z)
    return (; jx, jμ, jy, jz, n=kx + kμ + n_y + n_z)
end

# positions of the variables in v = (dx, dμ̃, dy, dz) (or (dx, dν̃, dκ, dz))
function _reiter_layout(lin::Linearization)
    N = size(lin.Π, 1)
    Nx = size(lin.Fx, 1)
    n_y = length(lin.aggregate_names); n_z = length(lin.exogenous_names)
    return merge((; N, Nx, n_y, n_z), _layout(Nx, N - 1, n_y, n_z))
end

# equilibrium conditions written one period earlier: no lead but predetermined aggregates
function _shifted_equations(lin::Linearization)
    ip = lin.predetermined
    shifted = [all(iszero, lin.AZ[a, :]) && all(iszero, lin.AY[a, .!ip]) && any(!iszero, lin.AY[a, ip])
               for a in axes(lin.AY, 1)]
    count(shifted) == count(ip) || error("Reiter: $(count(ip)) predetermined aggregate(s), but " *
        "$(count(shifted)) equilibrium condition(s) whose only leads are predetermined aggregates.")
    return shifted
end

"""
    reiter_system(lin) -> (; A, B, C, D, shifted)

Dense coefficients of `A E_t v_{t+1} + B v_t + C v_{t-1} + D ε_t = 0`,
`v = (dx, dμ̃, dy, dz)` (see `src/reiter.jl`).
"""
function reiter_system(lin::Linearization)
    L = _reiter_layout(lin)
    (; N, Nx, n_y, n_z, n, jx, jμ, jy, jz) = L
    shifted = _shifted_equations(lin)
    A = zeros(n, n); B = zeros(n, n); C = zeros(n, n); D = zeros(n, n_z)
    py, pz = lin.p_y, lin.p_z
    # dμ = Sμ dμ̃
    Sμ(M) = M[:, 1:N-1] .- M[:, N]                 # M dμ = (M Sμ) dμ̃
    # E1 (rows jx): Fx dx_t + FX dx_{t+1} + Fp dp_t + FP dp_{t+1} = 0
    r1 = jx
    B[r1, jx] = Matrix(lin.Fx); B[r1, jy] = lin.Fp * py; B[r1, jz] = lin.Fp * pz
    A[r1, jx] = Matrix(lin.FX); A[r1, jy] = lin.FP * py; A[r1, jz] = lin.FP * pz
    # E2 (rows jμ, first N-1 entries), at date t
    r2 = jμ
    Πt = Matrix(lin.Π')[1:N-1, :]
    B[r2, jμ] = Matrix(1.0I, N - 1, N - 1)
    B[r2, jy] = -(lin.ΠP * py)[1:N-1, :]; B[r2, jz] = -(lin.ΠP * pz)[1:N-1, :]
    C[r2, jμ] = -Sμ(Πt); C[r2, jx] = -Matrix(lin.Πx)[1:N-1, :]
    C[r2, jy] = -(lin.Πp * py)[1:N-1, :]; C[r2, jz] = -(lin.Πp * pz)[1:N-1, :]
    # E3 (rows jy): at date t, or t-1 for the shifted ones
    for a in 1:n_y
        r = jy[a]
        cur, lead = shifted[a] ? (C, B) : (B, A)
        cur[r, jμ] = Sμ(lin.Aμ[a:a, :]); cur[r, jx] = lin.Ax[a, :]
        cur[r, jy] = (lin.Ap * py + lin.Ay)[a, :]; cur[r, jz] = (lin.Ap * pz + lin.Az)[a, :]
        lead[r, jy] = lin.AY[a, :]; lead[r, jz] = lin.AZ[a, :]
    end
    # E4 (rows jz): dz_t = ρ dz_{t-1} + ε_t
    B[jz, jz] = Matrix(1.0I, n_z, n_z); C[jz, jz] = -lin.ρ; D[jz, :] = -Matrix(1.0I, n_z, n_z)
    return (; A, B, C, D, shifted)
end

# Minimal state v = (dx, dν̃, dκ, dz), all aggregates predetermined:
#   μ_t = ν_{t-1} + ΠP p_t,  y_t = κ_{t-1},  p_t = p_y κ_{t-1} + p_z z_t
#   (E1) Fx x_t + FX x_{t+1} + Fp p_t + FP p_{t+1} = 0
#   (ν)  ν_t = Π'ν_{t-1} + Πx x_t + (Π'ΠP + Πp) p_t          (first N-1 rows)
#   (E3) Aμ μ_t + Ax x_t + Ap p_t + Ay κ_{t-1} + AY κ_t + Az z_t = 0
function _minimal_system(lin::Linearization)
    all(lin.predetermined) || error("reduction=:minimal requires all aggregates to be predetermined")
    shifted = _shifted_equations(lin)               # checks AZ = 0 (all conditions are shifted)
    L = _reiter_layout(lin)
    (; N, n_y, n_z, n, jx, jμ, jy, jz) = L
    A = zeros(n, n); B = zeros(n, n); C = zeros(n, n); D = zeros(n, n_z)
    py, pz = lin.p_y, lin.p_z
    Sν(M) = M[:, 1:N-1] .- M[:, N]                  # M dν = (M Sν) dν̃
    add_p!(r, M) = (C[r, jy] .+= M * py; B[r, jz] .+= M * pz)    # M dp_t
    add_P!(r, M) = (B[r, jy] .+= M * py; A[r, jz] .+= M * pz)    # M dp_{t+1}
    # E1
    B[jx, jx] = Matrix(lin.Fx); A[jx, jx] = Matrix(lin.FX)
    add_p!(jx, lin.Fp); add_P!(jx, lin.FP)
    # ν
    Πt = Matrix(lin.Π')
    B[jμ, jμ] = Matrix(1.0I, N - 1, N - 1)
    C[jμ, jμ] = -Sν(Πt)[1:N-1, :]
    B[jμ, jx] = -Matrix(lin.Πx)[1:N-1, :]
    add_p!(jμ, -(Πt * lin.ΠP + lin.Πp)[1:N-1, :])
    # E3 (rows jy = κ)
    B[jy, jy] .+= lin.AY
    C[jy, jμ] .+= Sν(lin.Aμ)
    B[jy, jx] .+= lin.Ax
    add_p!(jy, lin.Aμ * lin.ΠP + lin.Ap)
    C[jy, jy] .+= lin.Ay; B[jy, jz] .+= lin.Az
    # E4
    B[jz, jz] = Matrix(1.0I, n_z, n_z); C[jz, jz] = -lin.ρ; D[jz, :] = -Matrix(1.0I, n_z, n_z)
    return (; A, B, C, D, shifted)
end

# --- Krylov bases ---------------------------------------------------------

# v ↦ -Fx⁻¹ FX v (matrix-free)
struct _PolicyOperator{F, S}
    Fxf::F
    FX::S
end
Base.:*(M::_PolicyOperator, v::AbstractVector) = -(M.Fxf \ (M.FX * v))

# Orthonormal basis V[:, 1:m] of the block Krylov space span{V0, M V0, M² V0, ...},
# extended on demand (block Arnoldi with double Gram-Schmidt and deflation of
# dependent vectors). MV[:, 1:nimg] = M V[:, 1:nimg].
mutable struct _KrylovBasis{O}
    M::O
    V::Matrix{Float64}
    MV::Matrix{Float64}
    m::Int
    nimg::Int
    queue::Vector{Vector{Float64}}  # vectors to orthogonalize next
    scale::Float64
    tol::Float64
    exhausted::Bool                 # invariant subspace found: the basis is exact
end

function _KrylovBasis(M, V0::AbstractMatrix; tol=1e-13)
    queue = [Vector{Float64}(V0[:, j]) for j in axes(V0, 2) if norm(V0[:, j]) > 0]
    scale = maximum(norm, queue; init=1.0)
    return _KrylovBasis(M, zeros(size(V0, 1), 0), zeros(size(V0, 1), 0), 0, 0, queue, scale, tol, isempty(queue))
end

function _reserve!(K::_KrylovBasis, k)
    N, cap = size(K.V)
    k <= cap && return
    cap = min(N, max(k, 2cap))
    V = zeros(N, cap); V[:, 1:K.m] = view(K.V, :, 1:K.m); K.V = V
    MV = zeros(N, cap); MV[:, 1:K.nimg] = view(K.MV, :, 1:K.nimg); K.MV = MV
end

function _next_image!(K::_KrylovBasis)
    K.nimg += 1
    w = K.M * K.V[:, K.nimg]
    K.MV[:, K.nimg] = w
    push!(K.queue, w)
end

# extends the basis to (at most) k vectors, and computes the images of all of them
function _extend!(K::_KrylovBasis, k::Int)
    k = min(k, size(K.V, 1))
    _reserve!(K, k)
    while K.m < k && !K.exhausted
        if isempty(K.queue)
            if K.nimg == K.m
                K.exhausted = true
                break
            end
            _next_image!(K)
        end
        w = popfirst!(K.queue)
        nw0 = norm(w)
        Vm = view(K.V, :, 1:K.m)
        for _ in 1:2
            h = Vm' * w
            mul!(w, Vm, h, -1.0, 1.0)
        end
        nw = norm(w)
        if nw > K.tol * max(nw0, K.scale) && nw > 1e-14
            K.m += 1
            K.V[:, K.m] .= w ./ nw
        end
    end
    while K.nimg < K.m
        _next_image!(K)
    end
    return K
end

_basis(K::_KrylovBasis) = K.V[:, 1:K.m]
_projection(K::_KrylovBasis) = (V = view(K.V, :, 1:K.m); V' * view(K.MV, :, 1:K.m))   # V'MV

# observability (distribution) and controllability (policy) Krylov bases
function _krylov_bases(lin::Linearization; tol=1e-13)
    μ = lin.steady_state.μ
    # Σ dμ = 0, so Aμ dμ = ã'dμ with centred rows
    ã = Matrix(lin.Aμ') .- ones(size(lin.Π, 1)) * (lin.Aμ * μ)'
    obs = _KrylovBasis(lin.Π, ã; tol)
    Fxf = lu(lin.Fx)
    gp = -(Fxf \ lin.Fp); gP = -(Fxf \ lin.FP)
    ctrl = _KrylovBasis(_PolicyOperator(Fxf, lin.FX), [gp gP]; tol)
    return (; obs, ctrl, ã, gp, gP)
end

# reduced system on v = (ξ, m, dy, dz), x = U ξ, m = V'dμ
function _krylov_system(lin::Linearization, kb)
    U = _basis(kb.ctrl); G = _projection(kb.ctrl)
    V = _basis(kb.obs); H = _projection(kb.obs)
    b_p = U' * kb.gp; b_P = U' * kb.gP
    c = V' * kb.ã                                      # Aμ dμ = c'm
    kx, kμ = size(U, 2), size(V, 2)
    n_y = length(lin.aggregate_names); n_z = length(lin.exogenous_names)
    L = _layout(kx, kμ, n_y, n_z)
    (; jx, jμ, jy, jz, n) = L
    A = zeros(n, n); B = zeros(n, n); C = zeros(n, n); D = zeros(n, n_z)
    py, pz = lin.p_y, lin.p_z
    # policy: ξ_t = G ξ_{t+1} + b_p dp_t + b_P dp_{t+1}
    B[jx, jx] = Matrix(1.0I, kx, kx); B[jx, jy] = -b_p * py; B[jx, jz] = -b_p * pz
    A[jx, jx] = -G; A[jx, jy] = -b_P * py; A[jx, jz] = -b_P * pz
    # distribution: m_t = H'm_{t-1} + V'(Πx U ξ_{t-1} + Πp dp_{t-1} + ΠP dp_t)
    VΠx = (lin.Πx' * V)' * U
    VΠp = V' * lin.Πp; VΠP = V' * lin.ΠP
    B[jμ, jμ] = Matrix(1.0I, kμ, kμ); B[jμ, jy] = -VΠP * py; B[jμ, jz] = -VΠP * pz
    C[jμ, jμ] = -H'; C[jμ, jx] = -VΠx; C[jμ, jy] = -VΠp * py; C[jμ, jz] = -VΠp * pz
    # equilibrium
    Ax = lin.Ax * U
    shifted = _shifted_equations(lin)
    for a in 1:n_y
        r = jy[a]
        cur, lead = shifted[a] ? (C, B) : (B, A)
        cur[r, jμ] = c[:, a]; cur[r, jx] = Ax[a, :]
        cur[r, jy] = (lin.Ap * py + lin.Ay)[a, :]; cur[r, jz] = (lin.Ap * pz + lin.Az)[a, :]
        lead[r, jy] = lin.AY[a, :]; lead[r, jz] = lin.AZ[a, :]
    end
    B[jz, jz] = Matrix(1.0I, n_z, n_z); C[jz, jz] = -lin.ρ; D[jz, :] = -Matrix(1.0I, n_z, n_z)
    return (; A, B, C, D, shifted, L, U, V)
end

# --- solution ---------------------------------------------------------------

function _solve_system(lin, sys, reduction, L; U=nothing, V=nothing, kwargs...)
    (; A, B, C, D, shifted) = sys
    cr = solve_cr(A, B, C; kwargs...)
    X = cr.X
    Y = -((A * X + B) \ D)
    N = size(lin.Π, 1)
    layout = (; jx=L.jx, jμ=L.jμ, jy=L.jy, jz=L.jz, n=L.n)
    return ReiterSolution(X, Y, cr, A, B, C, D, N, size(lin.Fx, 1) ÷ N, shifted, reduction, layout, U, V)
end

# responses of the aggregates to unit shocks (horizon × n_y × n_z)
function _unit_responses(sol::ReiterSolution, lin::Linearization, horizon)
    n_z = length(lin.exogenous_names)
    return cat((irf_reiter(sol, lin; T=horizon, shock=Matrix(1.0I, n_z, n_z)[:, j]).y for j in 1:n_z)...; dims=3)
end

"""
    solve_reiter(lin::Linearization; reduction=:auto, k=nothing, krylov_tol=1e-10, horizon=300, kwargs...) -> ReiterSolution

Solves the linearized model by Reiter's method: cyclic reduction on the dense
coefficients of the model in Dynare's form (see `src/reiter.jl`); `kwargs`
are passed to `solve_cr`. Errors if the Blanchard-Kahn conditions fail.

`reduction`:
- `:none`: the full system (controls, distribution, aggregates, exogenous);
- `:minimal`: exact, with the end-of-period distribution as state (only
  when all aggregates are predetermined); fewer lagged variables, a bit faster;
- `:auto` (default): `:minimal` when possible, `:none` otherwise;
- `:krylov`: two-sided Krylov reduction of the distribution and of the policy
  to (at most) `k` vectors each. With `k = nothing`, `k` grows (40, 60, 90, …)
  until the impulse responses of the aggregates to unit shocks (over
  `horizon` periods) change by less than `krylov_tol` (relative).
  Orders of magnitude faster on fine grids; approximate.
"""
function solve_reiter(lin::Linearization; reduction::Symbol=:auto, k=nothing, krylov_tol=1e-10,
                      horizon::Int=300, kwargs...)
    if reduction === :auto
        reduction = all(lin.predetermined) ? :minimal : :none
    end
    L = _reiter_layout(lin)
    reduction === :none && return _solve_system(lin, reiter_system(lin), :none, L; kwargs...)
    reduction === :minimal && return _solve_system(lin, _minimal_system(lin), :minimal, L; kwargs...)
    reduction === :krylov || error("unknown reduction $reduction (:auto, :none, :minimal or :krylov)")
    kb = _krylov_bases(lin)
    solve_k(k) = (_extend!(kb.obs, k); _extend!(kb.ctrl, k);
                  sys = _krylov_system(lin, kb);
                  _solve_system(lin, sys, :krylov, sys.L; U=sys.U, V=sys.V, kwargs...))
    k === nothing || return solve_k(k)
    kmax = max(size(lin.Π, 1), size(lin.Fx, 1))
    k = 40; prev = nothing
    while true
        sol = nothing; failure = nothing
        try
            sol = solve_k(k)
        catch err
            err isa ErrorException || rethrow()
            failure = err                              # too coarse (e.g. Blanchard-Kahn): refine
        end
        last = (kb.obs.exhausted && kb.ctrl.exhausted) || k >= kmax
        if failure !== nothing
            last && throw(failure)
        else
            r = _unit_responses(sol, lin, horizon)
            (last || (prev !== nothing && maximum(abs, r - prev) <= krylov_tol * maximum(abs, r))) && return sol
            prev = r
        end
        k = min(kmax, ceil(Int, 1.5k))
    end
end

# date-0 values of v after a shock dz_0 = shock, and the aggregates y_0
# (default: dμ_0 = ΠP dp_0)
# - :none, :krylov: v_0 = Y shock, or, if the date-0 distribution is given, the
#   solution of the date-0 equations (AX + B) v_0 = -D shock with the rows of
#   the distribution replaced by its coordinates (dμ̃_0 or V'dμ_0)
# - :minimal: v_0 = X v_{-1} + Y shock, with ν_{-1} = dμ_0 - ΠP dp_0 (y_0 = 0)
function _initial_state(sol::ReiterSolution, lin::Linearization, shock, dμ0)
    shock = collect(shock)
    dμ0 === nothing && return sol.Y * shock
    L = sol.layout; N = sol.N
    if sol.reduction === :minimal
        v = zeros(L.n)
        v[L.jμ] = (dμ0 - lin.ΠP * (lin.p_z * shock))[1:N-1]
        return sol.X * v + sol.Y * shock
    end
    M = sol.A * sol.X + sol.B
    rhs = -(sol.D * shock)
    M[L.jμ, :] .= 0.0
    M[L.jμ, L.jμ] = Matrix(1.0I, length(L.jμ), length(L.jμ))
    rhs[L.jμ] = sol.reduction === :krylov ? sol.V' * dμ0 : dμ0[1:N-1]
    return M \ rhs
end

# paths of all variables (n × T, dates 0..T-1) after a shock dz_0 = shock
function _reiter_paths(sol::ReiterSolution, lin::Linearization, T::Int, shock, dμ0=nothing)
    L = sol.layout
    p = _nzcols(sol.X)                      # variables appearing with a lag
    Xp = sol.X[:, p]
    V = zeros(L.n, T)
    V[:, 1] = _initial_state(sol, lin, shock, dμ0)
    for t in 2:T
        mul!(view(V, :, t), Xp, view(V, p, t - 1))
    end
    return V, L
end

# IRF from the paths of all variables (dy: n_y × T, dz: n_z × T)
function _make_irf(lin::Linearization, dy::AbstractMatrix, dz::AbstractMatrix)
    dp = lin.p_y * dy + lin.p_z * dz
    names = (; y=lin.aggregate_names, p=lin.price_names, z=lin.exogenous_names)
    return IRF(Matrix(dy'), Matrix(dp'), Matrix(dz'), names)
end

"""
    irf_reiter(sol::ReiterSolution, lin::Linearization; T=300, shock=_default_shock(lin), dμ0=nothing) -> IRF
    irf_reiter(lin::Linearization; T=300, shock=_default_shock(lin), dμ0=nothing, kwargs...)

Impulse response to a shock `dz_0 = shock` at date 0, dates 0..T-1. The
economy is at the steady state until date -1: predetermined aggregates are at
their steady-state values at date 0, and the distribution at date 0 already
responds to date-0 prices, dμ_0 = ΠP dp_0. The second form solves the model
first (`kwargs` are passed to `solve_reiter`).

`dμ0` overrides the deviation of the distribution at date 0 (it must sum to
zero; e.g. `zeros(N)` when date-0 prices should not move date-0 states).
"""
function irf_reiter(sol::ReiterSolution, lin::Linearization; T::Int=300, shock=_default_shock(lin), dμ0=nothing)
    V, L = _reiter_paths(sol, lin, T, shock, dμ0)
    # :minimal: y_t = κ_{t-1}, predetermined y_0 = 0
    dy = sol.reduction === :minimal ? [zeros(length(L.jy)) V[L.jy, 1:T-1]] : V[L.jy, :]
    return _make_irf(lin, dy, V[L.jz, :])
end

function irf_reiter(lin::Linearization; T::Int=300, shock=_default_shock(lin), dμ0=nothing, kwargs...)
    return irf_reiter(solve_reiter(lin; kwargs...), lin; T, shock, dμ0)
end

# appends the entries of M at offset (r0, c0)
function _push_block!(I, J, V, r0, c0, M)
    rows, cols, vals = findnz(sparse(M))
    append!(I, rows .+ r0); append!(J, cols .+ c0); append!(V, vals)
end

"""
    _irf_stacked(lin; T=300, shock=_default_shock(lin), dμ0=nothing)

Brute-force perfect-foresight response: E1-E4 stacked over dates 0..T (with
dμ_0 = ΠP dp_0 (or `dμ0`), dz_0 = shock, predetermined dy_0 = 0, and dx_T = 0, other dy_T = 0)
as one sparse linear system. Returns `(; irf, dx, dμ)` (dates 0..T-1, as
columns of dx and dμ). For tests: the truncation error is small far from T.
"""
function _irf_stacked(lin::Linearization; T::Int=300, shock=_default_shock(lin), dμ0=nothing)
    N = size(lin.Π, 1); Nx = size(lin.Fx, 1)
    n_y = length(lin.aggregate_names); n_z = length(lin.exogenous_names)
    m = Nx + N + n_y + n_z                     # unknowns per date: (dx, dμ, dy, dz)
    ox, oμ, oy, oz = 0, Nx, Nx + N, Nx + N + n_y
    c(t) = t * m                               # column offset of date t
    ip = findall(lin.predetermined); iu = findall(.!lin.predetermined)
    Fpy = lin.Fp * lin.p_y; Fpz = lin.Fp * lin.p_z
    FPy = lin.FP * lin.p_y; FPz = lin.FP * lin.p_z
    Πpy = lin.Πp * lin.p_y; Πpz = lin.Πp * lin.p_z
    ΠPy = lin.ΠP * lin.p_y; ΠPz = lin.ΠP * lin.p_z
    Apy = lin.Ap * lin.p_y; Apz = lin.Ap * lin.p_z
    Πt = sparse(lin.Π')
    I_, J_, V_ = Int[], Int[], Float64[]
    rhs = Float64[]
    r = 0
    # initial conditions: dμ_0 = ΠP dp_0 (or dμ0)
    _push_block!(I_, J_, V_, r, c(0) + oμ, sparse(1.0I, N, N))
    if dμ0 === nothing
        _push_block!(I_, J_, V_, r, c(0) + oy, -ΠPy)
        _push_block!(I_, J_, V_, r, c(0) + oz, -ΠPz)
    end
    r += N
    _push_block!(I_, J_, V_, r, c(0) + oz, sparse(1.0I, n_z, n_z)); r += n_z
    for (a, i) in enumerate(ip)
        push!(I_, r + a); push!(J_, c(0) + oy + i); push!(V_, 1.0)
    end
    r += length(ip)
    append!(rhs, dμ0 === nothing ? zeros(N) : dμ0, shock, zeros(length(ip)))
    for t in 0:T-1
        # E1
        _push_block!(I_, J_, V_, r, c(t) + ox, lin.Fx)
        _push_block!(I_, J_, V_, r, c(t) + oy, Fpy)
        _push_block!(I_, J_, V_, r, c(t) + oz, Fpz)
        _push_block!(I_, J_, V_, r, c(t+1) + ox, lin.FX)
        _push_block!(I_, J_, V_, r, c(t+1) + oy, FPy)
        _push_block!(I_, J_, V_, r, c(t+1) + oz, FPz)
        r += Nx
        # E2: dμ_{t+1} - Π'dμ_t - Πx dx_t - Πp dp_t - ΠP dp_{t+1} = 0
        _push_block!(I_, J_, V_, r, c(t+1) + oμ, sparse(1.0I, N, N))
        _push_block!(I_, J_, V_, r, c(t) + oμ, -Πt)
        _push_block!(I_, J_, V_, r, c(t) + ox, -lin.Πx)
        _push_block!(I_, J_, V_, r, c(t) + oy, -Πpy)
        _push_block!(I_, J_, V_, r, c(t) + oz, -Πpz)
        _push_block!(I_, J_, V_, r, c(t+1) + oy, -ΠPy)
        _push_block!(I_, J_, V_, r, c(t+1) + oz, -ΠPz)
        r += N
        # E3
        _push_block!(I_, J_, V_, r, c(t) + oμ, lin.Aμ)
        _push_block!(I_, J_, V_, r, c(t) + ox, lin.Ax)
        _push_block!(I_, J_, V_, r, c(t) + oy, Apy + lin.Ay)
        _push_block!(I_, J_, V_, r, c(t) + oz, Apz + lin.Az)
        _push_block!(I_, J_, V_, r, c(t+1) + oy, lin.AY)
        _push_block!(I_, J_, V_, r, c(t+1) + oz, lin.AZ)
        r += n_y
        # E4 (no shock after date 0)
        _push_block!(I_, J_, V_, r, c(t+1) + oz, sparse(1.0I, n_z, n_z))
        _push_block!(I_, J_, V_, r, c(t) + oz, -lin.ρ)
        r += n_z
        append!(rhs, zeros(Nx + N + n_y + n_z))
    end
    # terminal conditions
    _push_block!(I_, J_, V_, r, c(T) + ox, sparse(1.0I, Nx, Nx)); r += Nx
    for (a, i) in enumerate(iu)
        push!(I_, r + a); push!(J_, c(T) + oy + i); push!(V_, 1.0)
    end
    r += length(iu)
    append!(rhs, zeros(Nx + length(iu)))
    nv = (T + 1) * m
    r == nv || error("stacked system is not square ($r equations, $nv unknowns)")
    M = sparse(I_, J_, V_, nv, nv)
    v = reshape(M \ rhs, m, T + 1)[:, 1:T]
    irf = _make_irf(lin, v[oy+1:oy+n_y, :], v[oz+1:oz+n_z, :])
    return (; irf, dx=v[ox+1:ox+Nx, :], dμ=v[oμ+1:oμ+N, :])
end
