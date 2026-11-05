# Steady state as the solution of one system in u = (x, μ, y):
#
#     R_x = F(x, x; p(y))                    Euler equations on the grid
#     R_μ = Π(x, p(y))'μ - μ                 stationarity (last row replaced by Σμ - 1)
#     R_A = Σ_n μ_n e(s_n, x_n, y, z)        equilibrium conditions
#
# solved by an inexact Newton method, where each Newton step is computed with
# GMRES on the full linearized system (Newton-Krylov). The jacobian is never
# formed: its action is computed from the derivative operators of
# `derivatives.jl`. GMRES is right-preconditioned by a block lower triangular
# approximation of the jacobian:
#
#     agent:         zx = (I - L̃)⁻¹ F_x⁻¹ r_x   (truncated Neumann series, L̃ = -F_x⁻¹ F_X)
#     distribution:  zμ = M⁻¹ (r_μ - Π_x zx)     (M: I - Π' with the mass condition, factorized)
#     aggregates:    zy = S⁻¹ (r_A - A_x zx - A_μ zμ),  S = A_y + A_p p_y

struct JointPoint{H,DM,X,F}
    y::H          # aggregates (NamedTuple)
    dmodel::DM    # discretized agent, at prices p(y)
    x::X          # controls (GArray)
    φ::F          # decision rule interpolating x
    μ::Vector{Float64}
end

function _decision_rule(dmodel, x)
    vars = variables(dmodel.model.controls)
    φ = Dolo.DFun(dmodel.model.states, deepcopy(x), vars)
    Dolo.fit!(φ, x)
    return φ
end

function JointPoint(model::HModel, y::NamedTuple, z::NamedTuple, x::Dolo.GArray, μ::AbstractVector, discretization)
    dmodel = discretize(agent_model(model, y, z), discretization)
    x = Dolo.GArray(dmodel.grid, copy(x.data))
    JointPoint(y, dmodel, x, _decision_rule(dmodel, x), Vector{Float64}(μ))
end

# sizes of the blocks of u
function _sizes(pt::JointPoint)
    N = length(pt.x)
    n_x = length(pt.x[1])
    n_y = length(pt.y)
    return N, n_x, n_y
end

function _pack(x::Dolo.GArray, μ::AbstractVector, y::NamedTuple)
    vcat(_flatten(x), μ, collect(_as_vector(y)))
end

function _unpack(pt::JointPoint, u::AbstractVector)
    N, n_x, n_y = _sizes(pt)
    T = eltype(eltype(pt.x.data))
    x = Dolo.GArray(pt.x.grid, copy(reinterpret(SVector{n_x,T}, u[1:N*n_x])))
    μ = u[N*n_x+1:N*n_x+N]
    y = NamedTuple{keys(pt.y)}(Tuple(u[N*n_x+N+1:end]))
    return x, μ, y
end

"""
Residuals of the full system, as a flat vector `[R_x; R_μ; R_A]`. `pass` is
the `agent_pass` at `pt`, if already computed.
"""
function joint_residual(model::HModel, pt::JointPoint, z::NamedTuple; pass=nothing)
    (; dmodel, x, φ, μ, y) = pt
    if _sparse_jacobian_available(dmodel, φ)
        pass === nothing && (pass = agent_pass(dmodel, x; derivatives=false))
        R_x = _flatten(pass)
        R_μ = forward(pass, μ) - μ
    else
        R_x = _flatten(Dolo.F(dmodel, x, φ))
        R_μ = transition_matrix(dmodel, x)' * μ - μ
    end
    R_μ[end] = sum(μ) - 1
    R_A = aggregate(model, dmodel, μ, x, y, z)
    return vcat(R_x, R_μ, collect(R_A))
end

"""
Linearization of the full system at `pt`: returns `(Jv!, Pinv!, n)` where
`Jv!(w, v)` computes the action of the jacobian, `Pinv!(w, r)` the action of
the preconditioner, and `n` is the dimension of the system.
"""
function joint_linearization(model::HModel, pt::JointPoint, z::NamedTuple; agent_steps=0)

    (; dmodel, x, φ, μ, y) = pt
    N, n_x, n_y = _sizes(pt)
    names = price_names(model, y, z)

    p_y = projection_jacobian(model, y, z)
    (; J, L, F_p) = household_derivatives(dmodel, x, φ, names)
    Lt = precondition!(J, deepcopy(L))           # L̃ = -F_x⁻¹ F_X
    (; Π, Π_x, Π_p) = distribution_derivatives(dmodel, x, μ, names)
    (; A_μ, A_x, A_p, A_y) = aggregate_derivatives(model, dmodel, μ, x, y, z, names)

    # derivatives w.r.t. y through prices
    Fy = Dolo.GArray(dmodel.grid, [F_p[n] * p_y for n in 1:N])        # n_x × n_y blocks
    Π_y = Π_p * p_y                                                    # N × n_y
    S = A_y + A_p * p_y                                                # n_A × n_y

    # stationarity with the mass condition in the last row
    M = sparse(Π' - I)
    M = [M[1:end-1, :]; ones(1, N)]
    Π_x[end, :] .= 0.0; dropzeros!(Π_x)
    Π_y[end, :] .= 0.0
    Mf = lu(M)
    J_raw = J.data     # F_x (block diagonal)

    T = Float64
    ix = 1:N*n_x
    iμ = N*n_x+1:N*n_x+N
    iy = N*n_x+N+1:N*n_x+N+n_y

    tmp = zeros(T, N*n_x)

    function Jv!(w, v)
        vx = view(v, ix); vμ = view(v, iμ); vy = SVector{n_y}(v[iy])
        # agent
        mul!(tmp, L, collect(vx))
        vxg = reinterpret(SVector{n_x,T}, vx)
        wx = reinterpret(SVector{n_x,T}, view(w, ix))
        for n in 1:N
            wx[n] = J_raw[n] * vxg[n] + Fy[n] * vy
        end
        view(w, ix) .+= tmp
        # distribution
        view(w, iμ) .= M * vμ .+ Π_x * vx .+ Π_y * vy
        # aggregates
        view(w, iy) .= A_μ * vμ .+ A_x * vx .+ S * vy
        return w
    end

    function Pinv!(w, r)
        rx = view(r, ix); rμ = view(r, iμ); ry = r[iy]
        # agent: (I - L̃)⁻¹ F_x⁻¹ r_x, truncated
        rxg = reinterpret(SVector{n_x,T}, rx)
        zx = Dolo.GArray(dmodel.grid, [J.data[n] \ rxg[n] for n in 1:N])
        if agent_steps > 0
            term = deepcopy(zx)
            acc = deepcopy(zx)
            for _ in 1:agent_steps
                term = Lt * term
                acc.data .+= term.data
            end
            zx = acc
        end
        zxf = _flatten(zx)
        view(w, ix) .= zxf
        # distribution
        zμ = Mf \ (rμ .- Π_x * zxf)
        view(w, iμ) .= zμ
        # aggregates
        view(w, iy) .= S \ (ry .- A_μ * zμ .- A_x * zxf)
        return w
    end

    return Jv!, Pinv!, N*n_x + N + n_y
end

"""
    joint_newton_step(model, pt, z, R; distribution=nothing, pass=nothing)

Exact Newton step `du = -J⁻¹R` for the full system at `pt` (residuals `R`), by
block elimination, with the sparse jacobian of the agent (see `agent_jacobian`),
linear systems being solved by preconditioned GMRES (see `SparseSolver`):

    dx = X₀ + X₁ dy,   X₀ = -M⁻¹R_x,  X₁ = -M⁻¹F_y          (M = F_x + F_X)
    dμ = U₀ + U₁ dy                                          (stationarity system)
    (A_x X₁ + A_μ U₁ + S) dy = -R_A - A_x X₀ - A_μ U₀

`distribution` is the `DistributionSolver` of the previous step (the mode of
its distribution is used as normalization point), and `pass` the `agent_pass`
at `pt` with derivatives w.r.t. prices, if already computed. Returns
`(; du, distribution)`.
"""
function joint_newton_step(model::HModel, pt::JointPoint, z::NamedTuple, R::AbstractVector; distribution=nothing, pass=nothing)
    (; dmodel, x, φ, μ, y) = pt
    N, n_x, n_y = _sizes(pt)
    names = price_names(model, y, z)

    p_y = projection_jacobian(model, y, z)
    pass === nothing && (pass = agent_pass(dmodel, x, names))      # all derivatives, in one pass on the grid
    M = jacobian_matrix(pass)
    F_p = pass.Fp
    (; Π_x, Π_p) = distribution_derivatives(pass, μ; Π=false)
    (; A_μ, A_x, A_p, A_y) = aggregate_derivatives(model, dmodel, μ, x, y, z, names)

    Fy = _price_effect(F_p, p_y, n_x, n_y)     # F_p p_y (flattened)
    Π_y = Π_p * p_y
    S = A_y + A_p * p_y

    R_x = view(R, 1:N*n_x)
    R_μ = view(R, N*n_x+1:N*n_x+N)
    R_A = view(R, N*n_x+N+1:N*n_x+N+n_y)

    # agent
    Mf = SparseSolver(M; rtol=1e-6)    # preconditioned GMRES (inexact Newton)
    X0 = -(Mf \ collect(R_x))
    X1 = -(Mf \ Fy)

    # distribution: rows 1..N-1 are (Π' - I)dμ + Π_x dx + Π_y dy = -R_μ, the last one Σdμ = -R_μ[N]
    D = _distribution_factorization(pass, distribution; rtol=1e-6, guess=μ)
    U0 = _solve_mass(D, R_μ .+ Π_x * X0, -R_μ[end])
    U1 = reduce(hcat, [_solve_mass(D, Π_x * X1[:, c] .+ Π_y[:, c], 0.0) for c in 1:n_y])

    # aggregates
    dy = (A_x * X1 + A_μ * U1 + S) \ (-R_A - A_x * X0 - A_μ * U0)
    return (; du=vcat(X0 + X1 * dy, U0 + U1 * dy, dy), distribution=D)
end

# rows (n-1)n_x+1 : n n_x of the result are F_p[n] p_y
function _price_effect(F_p::AbstractVector, p_y, n_x, n_y)
    N = length(F_p)
    out = zeros(N * n_x, n_y)
    @inbounds for n in 1:N
        B = F_p[n] * p_y
        for c in 1:n_y, a in 1:n_x
            out[(n-1)*n_x+a, c] = B[a, c]
        end
    end
    return out
end

# Initial point (x, μ) at y0. If the agent defines a `cold_start` (e.g. `egm`),
# its decision rule is used as is: it does not solve exactly the agent's
# problem on Dolo's grid, but Newton-Krylov iterations take care of that.
# Otherwise the agent's problem is solved accurately.
function _initial_point(model::HModel, y0, z, discretization)
    dmodel = discretize(agent_model(model, y0, z), discretization)
    x0 = cold_start(dmodel.model, dmodel)
    if x0 === nothing
        x0 = solve_agent(dmodel).dr.values
    end
    μ0 = ergodic_distribution(dmodel, x0)
    return JointPoint(model, y0, z, x0, μ0, discretization)
end

"""
    solve_steady_state_joint(model; y0=calibrated, z=calibrated, discretization=Dict(),
        tol=1e-8, maxit=30, agent_steps=0, krylov_dim=100, η=1e-4, verbose=true)

Solves for the steady state with a Newton-Krylov method on the full system
`(x, μ, y)`. The initial guess for `x` and `μ` at `y0` comes from the agent's
`cold_start` if it defines one (e.g. `egm`), otherwise from solving the
agent's problem. Each Newton step is computed with right-preconditioned
GMRES (see `joint_linearization`), to a relative tolerance `η`, or, with
`linsolve=:direct` (the default when the agent's sparse jacobian is available,
see `agent_jacobian`), exactly, by block elimination (see `joint_newton_step`).
`agent_steps` is the number of Neumann terms in the agent's preconditioner for
GMRES (0: only the block diagonal `F_x⁻¹`). Each Newton step is scaled so that
aggregates change by at most `max_step_y` (relative).

Returns `(; y, z, A, dmodel, x, μ, iterations, gmres_iterations)`.
"""
function solve_steady_state_joint(model::HModel;
        y0::NamedTuple=calibrated_aggregate(model),
        z::NamedTuple=calibrated_exogenous(model),
        discretization=Dict(),
        tol=1e-8,
        maxit=30,
        agent_steps=0,
        krylov_dim=100,
        max_restarts=20,
        η=1e-4,
        max_backsteps=10,
        max_step_y=0.2,
        linsolve=:auto,
        verbose=true)

    pt = _initial_point(model, y0, z, discretization)
    R = joint_residual(model, pt, z)
    ε = norm(R, Inf)
    N, n_x, n_y = _sizes(pt)
    yvars = keys(y0)
    if linsolve == :auto
        linsolve = _sparse_jacobian_available(pt.dmodel, pt.φ) ? :direct : :gmres
    end
    linsolve in (:direct, :gmres) || error("Unknown linsolve $linsolve: use :direct or :gmres.")
    distribution = nothing
    pass = nothing          # agent pass (with derivatives) at the current point

    verbose && println("Steady state (Newton-Krylov): solving for x, μ, (", join(yvars, ", "), ")")
    verbose && println(" it | ", join(rpad.(string.(yvars), 12), " "), " | residual     | gmres | backsteps")

    gmres_its = Int[]
    it = 0
    k_lin = 0
    k_back = 0
    while true
        verbose && println(lpad(it, 3), " | ", join((rpad(round(v; sigdigits=8), 12) for v in values(pt.y)), " "),
            " | ", rpad(round(ε; sigdigits=4), 12), " | ", lpad(k_lin, 5), " | ", lpad(k_back, 3))
        (ε < tol || it >= maxit) && break
        it += 1

        n = length(R)
        if linsolve == :direct
            (; du, distribution) = joint_newton_step(model, pt, z, R; distribution=distribution, pass=pass)
            k_lin = 0
        else
            Jv!, Pinv!, n = joint_linearization(model, pt, z; agent_steps=agent_steps)

            # right preconditioning: (J P⁻¹) w = -R, du = P⁻¹ w
            aux = zeros(n)
            A! = (out, v) -> (Pinv!(aux, v); Jv!(out, aux))
            wk = Dolo.gmres_workspace(Float64, n, krylov_dim)
            w = zeros(n)
            b = -R
            tol_lin = η * norm(b)
            k_lin = 0
            for _ in 1:max_restarts
                k, converged = Dolo.gmres!(w, A!, b, wk; tol=tol_lin)
                k_lin += k
                converged && break
            end
            du = Pinv!(zeros(n), w)
        end
        push!(gmres_its, k_lin)

        # limit the relative change of aggregates (the full step can leave the
        # region where the agent's problem is well behaved)
        dy = view(du, N*n_x+N+1:n)
        yv = _as_vector(pt.y)
        ρ = maximum(abs.(dy) ./ (max_step_y .* max.(abs.(yv), 1.0)))
        ρ > 1 && (du ./= ρ)

        # backtracking on the residual
        u0 = _pack(pt.x, pt.μ, pt.y)
        accepted = false
        for k in 0:max_backsteps
            x1, μ1, y1 = _unpack(pt, u0 .+ du .* 0.5^k)
            y1 = _as_namedtuple(model, _clamp(model, _as_vector(y1)))
            pt1, R1, ε1, pass1 = try
                pt1 = JointPoint(model, y1, z, x1, μ1, discretization)
                # with derivatives: reused for the next Newton step if the point is accepted
                pass1 = linsolve == :direct ? agent_pass(pt1.dmodel, pt1.x, price_names(model, y1, z)) : nothing
                R1 = joint_residual(model, pt1, z; pass=pass1)
                pt1, R1, norm(R1, Inf), pass1
            catch err
                err isa DomainError || rethrow()
                nothing, nothing, Inf, nothing     # e.g. negative consumption
            end
            if isfinite(ε1) && ε1 < ε
                pt, R, ε, pass = pt1, R1, ε1, pass1
                k_back = k
                accepted = true
                break
            end
        end
        if !accepted
            @warn "No step decreases the residual: stopping."
            break
        end
    end

    ε < tol || @warn "Steady state did not converge (residual $ε)"

    A = aggregate(model, pt.dmodel, pt.μ, pt.x, pt.y, z)
    return (; y=pt.y, z, A, dmodel=pt.dmodel, x=pt.x, μ=pt.μ, iterations=it, gmres_iterations=gmres_its)
end
