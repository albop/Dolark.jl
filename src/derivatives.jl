# Derivatives of the steady-state conditions.
#
# Given aggregates y (and z), prices are p = projection(y, z). The steady state of
# the agents' block is characterized by
#
#     F(x, x; p) = 0                    (household: Euler equations on the grid)
#     μ = Π(x, p)' μ,  Σμ = 1           (stationary distribution)
#
# and the equilibrium conditions are
#
#     A(μ, x, p, y) = Σ_n μ_n e(s_n(p), x_n, y, z, y, z) = 0
#
# By the implicit function theorem:
#
#     dx/dy = -(F_x + F_X)⁻¹ F_p p_y
#     (I - Π') dμ/dy = Π'_x dx/dy + Π'_p p_y,       Σ dμ = 0
#     dA/dy = A_μ dμ/dy + A_x dx/dy + A_p p_y + A_y
#
# where Π'_x (resp. Π'_p) is the derivative of v ↦ Π(x,p)'μ w.r.t. x (resp. p).
#
# Prices are assumed to enter the agent's problem only through the values of the
# exogenous states (not through the agent's calibration).

using ForwardDiff
import Dolo: QP

###
### Prices as vectors
###

price_names(model::HModel, y=calibrated_aggregate(model), z=calibrated_exogenous(model)) = keys(projection(model, y, z))

function projection_vector(model::HModel, y::NamedTuple, z::NamedTuple)
    SVector(values(projection(model, y, z))...)
end

# p_y: jacobian of prices w.r.t. aggregates
function projection_jacobian(model::HModel, y::NamedTuple, z::NamedTuple)
    vars = keys(y)
    ForwardDiff.jacobian(
        u -> projection_vector(model, NamedTuple{vars}(Tuple(u)), z),
        _as_vector(y)
    )
end

###
### Pointwise building blocks, generic in the number type of x and p
###

# values of the exogenous states, where variables in `p` are replaced
@inline function _exo_values(vars::NTuple{k,Symbol}, q::SVector{k}, p::NamedTuple) where k
    T = promote_type(eltype(q), map(typeof, values(p))...)
    SVector{k,T}(ntuple(i -> haskey(p, vars[i]) ? convert(T, p[vars[i]]) : convert(T, q[i]), Val(k)))
end

@inline function _endo_part(v::SVector{d}, ::Val{k}) where d where k
    SVector{d-k}(ntuple(i -> v[k+i], Val(d-k)))
end

# current state at grid point n, with prices p
@inline function _state(dmodel::Dolo.DYModel, n::Int, p::NamedTuple)
    vars = variables(dmodel.model.exogenous)
    ss = Dolo.point(dmodel.grid, n)
    i = ss.loc[1]
    m = _exo_values(vars, dmodel.dproc.Q[i], p)
    s_endo = _endo_part(ss.val, Val(length(vars)))
    T = promote_type(eltype(m), eltype(s_endo))
    return i, QP(ss.loc, SVector{length(ss.val),T}(m..., s_endo...))
end

# next period's states, for each exogenous node j: (P[i,j], S::QP)
@inline function _future_states(dmodel::Dolo.DYModel, i::Int, s::QP, x::SVector, p::NamedTuple)
    vars = variables(dmodel.model.exogenous)
    P = dmodel.dproc.P
    Q = dmodel.dproc.Q
    return (
        let
            M = _exo_values(vars, Q[j], p)
            S_endo = SVector(Dolo.transition(dmodel.model, s.val, x, M)...)
            (P[i, j], QP((j, S_endo), SVector(M..., S_endo...)))
        end
        for j in 1:size(P, 2)
    )
end

"""
Residual of the Euler equations at grid point `n`, for controls `x_n` today, a
decision rule `φ` tomorrow, and prices `p`. Same as `Dolo.F` when `p` is the
current value of prices.
"""
function household_residual(dmodel::Dolo.DYModel, n::Int, x_n::SVector, φ, p::NamedTuple)
    i, s = _state(dmodel, n, p)
    T = promote_type(eltype(x_n), eltype(s.val))
    r = zero(SVector{length(x_n),T})
    for (w, S) in _future_states(dmodel, i, s, x_n, p)
        r += w * Dolo.arbitrage(dmodel, s, x_n, S, φ(S))
    end
    return Dolo.complementarities(dmodel.model, s, x_n, r)
end

# Lottery on a 1d regular grid: S is split between nodes k and k+1 with weights
# (1-λ, λ). Outside of the grid, all the mass goes to the closest bound.
@inline function _lottery(g::Dolo.CGrid{1}, S::Real)
    a, b, ng = g.ranges[1]
    u = (S - a) / (b - a) * (ng - 1)
    k = floor(Int, ForwardDiff.value(u))
    k = max(min(k, ng - 2), 0)
    λ = u - k
    λ = max(min(λ, one(λ)), zero(λ))
    return k + 1, λ
end

"""
Transitions from grid point `n`, for controls `x_n` and prices `p`: a tuple of
`(j, k, w)` meaning that a mass `w` moves to point (j, k) of the grid, where `j`
is the exogenous index and `k` the endogenous one.
"""
function point_transitions(dmodel::Dolo.DYModel, n::Int, x_n::SVector, p::NamedTuple)
    endo_grid = dmodel.grid.grids[2]
    i, s = _state(dmodel, n, p)
    return Iterators.flatten(
        let
            j, S_endo = S.loc
            k, λ = _lottery(endo_grid, S_endo[1])
            ((j, k, w * (1 - λ)), (j, k + 1, w * λ))
        end
        for (w, S) in _future_states(dmodel, i, s, x_n, p)
    )
end

###
### Derivative blocks
###

_current_prices(dmodel::Dolo.DYModel, names) = NamedTuple{names}(map(k -> dmodel.model.calibration[k], names))

"""
    household_derivatives(dmodel, x, φ, names)

Returns `(; J, L, F_p)` with `J = F_x` (block diagonal, a GArray of matrices),
`L = F_X` (a Dolo linear operator) and `F_p` (a GArray of `n_x × n_p` matrices,
derivative w.r.t. prices `names`).
"""
function household_derivatives(dmodel::Dolo.DYModel, x::Dolo.GArray, φ, names::Tuple)
    p0 = _current_prices(dmodel, names)
    pv = _as_vector(p0)
    lφ = Dolo.Linearized(φ)
    F_p = Dolo.GArray(dmodel.grid, [
        ForwardDiff.jacobian(u -> household_residual(dmodel, n, x[n], lφ, NamedTuple{names}(Tuple(u))), pv)
        for n in 1:length(x)
    ])
    J = Dolo.dF_1(dmodel, x, φ)
    L = Dolo.dF_2(dmodel, x, φ)
    return (; J, L, F_p)
end

"""
Solves `(F_x + F_X) dx = b` (with the same preconditioning as `Dolo.newton`):
`(I - L) dx = F_x⁻¹ b`, by GMRES. `L` must already be replaced by
`-F_x⁻¹ F_X`, see `precondition!`. Restarted every `krylov_dim` iterations.
"""
function solve_household(J, L, b::Dolo.GArray; tol=1e-12, krylov_dim=50, max_restarts=20)
    r = Dolo.GArray(b.grid, J.data .\ b.data)
    rv = Dolo.ravel(r)
    T = eltype(rv)
    wk = Dolo.gmres_workspace(T, length(rv), krylov_dim)
    dx = copy(rv)
    A! = (w, v) -> (mul!(w, L, v); w .= v .- w)
    for _ in 1:max_restarts    # restarted GMRES
        _, converged = Dolo.gmres!(dx, A!, rv, wk; tol=tol)
        converged && break
    end
    return Dolo.unravel(r, dx)
end

precondition!(J, L) = (ldiv!(J, L; α=-1.0); L)

"""
    distribution_derivatives(dmodel, x, μ, names)

Returns `(; Π, Π_x, Π_p)` where `Π` is the transition matrix, and `Π_x`
(sparse, `N × N n_x`) and `Π_p` (`N × n_p`) are the derivatives of `Π(x,p)'μ`
w.r.t. `x` (flattened) and prices.
"""
function distribution_derivatives(dmodel::Dolo.DYModel, x::Dolo.GArray, μ::AbstractVector, names::Tuple)
    grid = dmodel.grid
    n_m, n_s = size(grid)
    N = n_m * n_s
    n_x = length(x[1])
    n_p = length(names)
    p0 = _current_prices(dmodel, names)
    pv = _as_vector(p0)

    Ip, Jp, Vp = Int[], Int[], Float64[]      # Π
    Ix, Jx, Vx = Int[], Int[], Float64[]      # Π_x
    Π_p = zeros(N, n_p)

    for n in 1:N
        x_n = x[n]
        targets = [ _linear_index(n_m, j, k) for (j, k, _) in point_transitions(dmodel, n, x_n, p0) ]
        weights(u) = SVector((w for (_, _, w) in point_transitions(dmodel, n,
                SVector{n_x}(ntuple(l -> u[l], Val(n_x))),
                NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p))) ))...)
        u0 = vcat(x_n, pv)
        w0 = weights(u0)
        D = ForwardDiff.jacobian(weights, u0)
        for (r, t) in enumerate(targets)
            push!(Ip, n); push!(Jp, t); push!(Vp, w0[r])
            for l in 1:n_x
                push!(Ix, t); push!(Jx, (n - 1) * n_x + l); push!(Vx, μ[n] * D[r, l])
            end
            for l in 1:n_p
                Π_p[t, l] += μ[n] * D[r, n_x + l]
            end
        end
    end

    Π = sparse(Ip, Jp, Vp, N, N)
    Π_x = sparse(Ix, Jx, Vx, N, N * n_x)
    return (; Π, Π_x, Π_p)
end

"""
    aggregate_derivatives(model, dmodel, μ, x, y, z, names)

Returns `(; A, A_μ, A_x, A_p, A_y)`: the aggregated equilibrium conditions and
their derivatives w.r.t. the distribution (`n_A × N`), the controls (`n_A × N n_x`),
the prices (`n_A × n_p`) and the aggregates (`n_A × n_y`, current and future).
"""
function aggregate_derivatives(model::HModel, dmodel::Dolo.DYModel, μ::AbstractVector, x::Dolo.GArray, y::NamedTuple, z::NamedTuple, names::Tuple)
    N = length(x)
    n_x = length(x[1])
    n_p = length(names)
    yvars = keys(y)
    n_y = length(yvars)
    p0 = _current_prices(dmodel, names)
    u0 = vcat(x[1], _as_vector(p0), _as_vector(y))

    function e(n, u)
        x_n = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
        p = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
        yy = NamedTuple{yvars}(ntuple(l -> u[n_x+n_p+l], Val(n_y)))
        _, s = _state(dmodel, n, p)
        equilibrium(model, s.val, x_n, yy, z, yy, z)
    end

    n_A = length(e(1, u0))
    A = zeros(n_A)
    A_μ = zeros(n_A, N)
    A_x = zeros(n_A, N * n_x)
    A_p = zeros(n_A, n_p)
    A_y = zeros(n_A, n_y)
    for n in 1:N
        u = vcat(x[n], _as_vector(p0), _as_vector(y))
        en = e(n, u)
        D = ForwardDiff.jacobian(v -> e(n, v), u)
        A .+= μ[n] .* en
        A_μ[:, n] .= en
        A_x[:, (n-1)*n_x+1:n*n_x] .= μ[n] .* D[:, 1:n_x]
        A_p .+= μ[n] .* D[:, n_x+1:n_x+n_p]
        A_y .+= μ[n] .* D[:, n_x+n_p+1:end]
    end
    return (; A, A_μ, A_x, A_p, A_y)
end

###
### Total derivative
###

# solves (I-Π')dμ = r with Σdμ = 0
function _solve_distribution(Π, r::AbstractVector)
    N = size(Π, 1)
    M = sparse(I - Π')
    M = [M[1:end-1, :]; ones(1, N)]
    return M \ vcat(r[1:end-1], 0.0)
end

_flatten(x::Dolo.GArray) = collect(reinterpret(eltype(eltype(x.data)), x.data))

"""
    excess_jacobian(model, res; z=calibrated)

Jacobian `dA/dy` of the aggregated equilibrium conditions w.r.t. the aggregates,
at the point described by `res` (the output of `excess`). Computed exactly, by
the implicit function theorem.
"""
function excess_jacobian(model::HModel, res; z::NamedTuple=calibrated_exogenous(model))
    (; dmodel, sol, x, μ, y) = res
    φ = sol.dr
    names = price_names(model, y, z)

    p_y = projection_jacobian(model, y, z)
    (; J, L, F_p) = household_derivatives(dmodel, x, φ, names)
    precondition!(J, L)
    (; Π, Π_x, Π_p) = distribution_derivatives(dmodel, x, μ, names)
    (; A_μ, A_x, A_p, A_y) = aggregate_derivatives(model, dmodel, μ, x, y, z, names)

    n_y = length(y)
    dA = zeros(size(A_y, 1), n_y)
    for c in 1:n_y
        dp = p_y[:, c]
        b = Dolo.GArray(dmodel.grid, [-(F_p[n] * dp) for n in 1:length(x)])
        dx = _flatten(solve_household(J, L, b))
        dμ = _solve_distribution(Π, Π_x * dx + Π_p * dp)
        dA[:, c] = A_μ * dμ + A_x * dx + A_p * dp + A_y[:, c]
    end
    return dA
end
