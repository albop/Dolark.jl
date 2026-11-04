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

"Residuals of the full system, as a flat vector `[R_x; R_μ; R_A]`."
function joint_residual(model::HModel, pt::JointPoint, z::NamedTuple)
    (; dmodel, x, φ, μ, y) = pt
    R_x = _flatten(Dolo.F(dmodel, x, φ))
    Π = transition_matrix(dmodel, x)
    R_μ = Π' * μ - μ
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
    solve_steady_state_joint(model; y0=calibrated, z=calibrated, discretization=Dict(),
        tol=1e-8, maxit=30, agent_steps=0, krylov_dim=100, η=1e-4, verbose=true)

Solves for the steady state with a Newton-Krylov method on the full system
`(x, μ, y)`. The agent's problem is first solved at `y0`, to get an initial
guess for `x` and `μ`. Each Newton step is computed with right-preconditioned
GMRES (see `joint_linearization`), to a relative tolerance `η`.
`agent_steps` is the number of Neumann terms in the agent's preconditioner
(0: only the block diagonal `F_x⁻¹`). Each Newton step is scaled so that
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
        verbose=true)

    res0 = excess(model, y0; z=z, discretization=discretization)
    pt = JointPoint(model, y0, z, res0.x, res0.μ, discretization)
    R = joint_residual(model, pt, z)
    ε = norm(R, Inf)
    N, n_x, n_y = _sizes(pt)
    yvars = keys(y0)

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
        push!(gmres_its, k_lin)
        du = Pinv!(zeros(n), w)

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
            pt1, R1, ε1 = try
                pt1 = JointPoint(model, y1, z, x1, μ1, discretization)
                R1 = joint_residual(model, pt1, z)
                pt1, R1, norm(R1, Inf)
            catch err
                err isa DomainError || rethrow()
                nothing, nothing, Inf     # e.g. negative consumption
            end
            if isfinite(ε1) && ε1 < ε
                pt, R, ε = pt1, R1, ε1
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
