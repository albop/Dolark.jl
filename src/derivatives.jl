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
# names of the exogenous variables, as a type parameter
@inline _exo_names(dmodel::Dolo.DYModel{<:Any,<:Any,<:Dolo.MarkovChain{names}}) where names = Val(names)
@inline _exo_names(dmodel::Dolo.DYModel) = Val(variables(dmodel.model.exogenous))
@inline _exo_values(vars::NTuple{k,Symbol}, q::SVector{k}, p::NamedTuple) where k = _exo_values(Val(vars), q, p)

# (names are matched at compile time: `haskey` on a NamedTuple of dual numbers allocates)
@generated function _exo_values(::Val{vars}, q::SVector{k,Tq}, p::NamedTuple{pn,Tp}) where {vars, k, Tq, pn, Tp}
    T = promote_type(Tq, Tp.parameters...)
    entries = [v in pn ? :(convert($T, p.$v)) : :(convert($T, q[$i])) for (i, v) in enumerate(vars)]
    return :(SVector{$k,$T}($(entries...)))
end

@inline function _endo_part(v::SVector{d}, ::Val{k}) where d where k
    SVector{d-k}(ntuple(i -> v[k+i], Val(d-k)))
end

# current state at grid point n, with prices p
@inline function _state(dmodel::Dolo.DYModel, n::Int, p::NamedTuple)
    vars = variables(dmodel.model.exogenous)
    ss = Dolo.point(dmodel.grid, n)
    i = ss.loc[1]
    m = _exo_values(_exo_names(dmodel), dmodel.dproc.Q[i], p)
    s_endo = _endo_part(ss.val, Val(length(vars)))
    T = promote_type(eltype(m), eltype(s_endo))
    return i, QP(ss.loc, SVector{length(ss.val),T}(m..., s_endo...))
end

# number of exogenous nodes, as a type parameter
_n_nodes(dmodel::Dolo.DYModel) = Val(size(dmodel.dproc.P, 2))

# next period's states, for each exogenous node j: a tuple of (P[i,j], S::QP)
@inline function _future_states(dmodel::Dolo.DYModel, i::Int, s::QP, x::SVector, p::NamedTuple)
    _future_states(dmodel, i, s, x, p, _n_nodes(dmodel))
end

@inline function _future_states(dmodel::Dolo.DYModel, i::Int, s::QP, x::SVector, p::NamedTuple, ::Val{K}) where K
    P = dmodel.dproc.P
    Q = dmodel.dproc.Q
    return ntuple(Val(K)) do j
        M = _exo_values(_exo_names(dmodel), Q[j], p)
        S_endo = SVector(Dolo.transition(dmodel.model, s.val, x, M)...)
        (P[i, j], QP((j, S_endo), SVector(M..., S_endo...)))
    end
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

# One-dimensional endogenous grids: uniform `CGrid{1}`, or any 1d grid of Dolo
# (`Uniform1D`, `LogGrid`, `AssetGrid`, `PowerGrid`, ...) wrapped in a `CartesianGrid{1}`.
_grid1d(g::Dolo.CGrid{1}) = g
_grid1d(g::Dolo.CartesianGrid{1}) = g.grids[1]
_grid1d(g) = error("Only 1d endogenous grids are supported for now (got $(typeof(g))).")

_bounds(g::Dolo.CGrid{1}) = g.ranges[1]
_bounds(g::Dolo.AGrid1D) = (g[1], g[length(g)], length(g))

# Lottery on a 1d grid: S is split between nodes k and k+1 with weights
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

@inline function _lottery(g::Dolo.AGrid1D, S::Real)
    # cell located from the value; λ keeps the derivative w.r.t. S
    k, _ = Dolo.locate_cell(g, ForwardDiff.value(S))
    λ = (S - g[k]) / (g[k+1] - g[k])
    λ = max(min(λ, one(λ)), zero(λ))
    return k, λ
end

"""
Lotteries from grid point `n`, for controls `x_n` and prices `p`: a tuple with,
for each future exogenous node `j`, `(j, k, w, λ)` meaning that a mass
`w (1-λ)` moves to point `(j, k)` of the grid and a mass `w λ` to `(j, k+1)`.
"""
@inline function point_lotteries(dmodel::Dolo.DYModel, n::Int, x_n::SVector, p::NamedTuple)
    endo_grid = _grid1d(dmodel.grid.grids[2])
    i, s = _state(dmodel, n, p)
    return map(_future_states(dmodel, i, s, x_n, p)) do (w, S)
        j, S_endo = S.loc
        k, λ = _lottery(endo_grid, S_endo[1])
        (j, k, w, λ)
    end
end

"""
Transitions from grid point `n`, for controls `x_n` and prices `p`: a tuple of
`(j, k, w)` meaning that a mass `w` moves to point (j, k) of the grid, where `j`
is the exogenous index and `k` the endogenous one.
"""
function point_transitions(dmodel::Dolo.DYModel, n::Int, x_n::SVector, p::NamedTuple)
    return Iterators.flatten(
        ((j, k, w * (1 - λ)), (j, k + 1, w * λ))
        for (j, k, w, λ) in point_lotteries(dmodel, n, x_n, p)
    )
end

###
### Derivative blocks
###

_current_prices(dmodel::Dolo.DYModel, names) = NamedTuple{names}(map(k -> dmodel.model.calibration[k], names))

"""
    price_derivatives(dmodel, x, φ, names)

`F_p`: derivative of the agent's residuals w.r.t. prices `names` (a GArray of
`n_x × n_p` matrices).
"""
function price_derivatives(dmodel::Dolo.DYModel, x::Dolo.GArray, φ, names::Tuple)
    _price_derivatives(dmodel, x, Dolo.Linearized(φ), Val(names))
end

function _price_derivatives(dmodel, x, lφ, ::Val{names}) where names
    pv = _as_vector(_current_prices(dmodel, names))
    f(n, u) = household_residual(dmodel, n, x[n], lφ, NamedTuple{names}(Tuple(u)))
    return Dolo.GArray(dmodel.grid, [ForwardDiff.jacobian(u -> f(n, u), pv) for n in 1:length(x)])
end

"""
    household_derivatives(dmodel, x, φ, names)

Returns `(; J, L, F_p)` with `J = F_x` (block diagonal, a GArray of matrices),
`L = F_X` (a Dolo linear operator) and `F_p` (a GArray of `n_x × n_p` matrices,
derivative w.r.t. prices `names`).
"""
function household_derivatives(dmodel::Dolo.DYModel, x::Dolo.GArray, φ, names::Tuple)
    F_p = price_derivatives(dmodel, x, φ, names)
    J = Dolo.dF_1(dmodel, x, φ)
    # as `Dolo.dF_2(dmodel, x, φ)`, with `has_complementarities` (reflection) evaluated once
    vc = Val(Dolo.has_complementarities(dmodel.model))
    L = Dolo.LL(dmodel.grid, [Dolo.dF_2_kernel(n, dmodel, x, φ, vc) for n in 1:length(x)], deepcopy(φ))
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
    # (sizes and names as type parameters: function barrier)
    _distribution_derivatives(dmodel, x, μ, Val(names), Val(length(eltype(x.data))), _n_nodes(dmodel))
end

# lottery weights λ_j at point n, as a function of u = (x_n, p)
@inline function _lottery_weights(dmodel, n, u::SVector, names::NTuple{n_p,Symbol}, ::Val{n_x}) where {n_p, n_x}
    x_n = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
    p = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
    return SVector(map(t -> t[4], point_lotteries(dmodel, n, x_n, p)))
end

function _distribution_derivatives(dmodel, x, μ, ::Val{names}, ::Val{n_x}, ::Val{K}) where {names, n_x, K}
    grid = dmodel.grid
    n_m, n_s = size(grid)
    N = n_m * n_s
    n_p = length(names)
    p0 = _current_prices(dmodel, names)
    pv = _as_vector(p0)

    Ip = Vector{Int}(undef, 2K * N); Jp = similar(Ip); Vp = Vector{Float64}(undef, 2K * N)
    Ix = Vector{Int}(undef, 2K * N * n_x); Jx = similar(Ix); Vx = Vector{Float64}(undef, 2K * N * n_x)
    Π_p = zeros(N, n_p)

    cp = 0
    cx = 0
    @inbounds for n in 1:N
        u0 = vcat(x[n], pv)
        lot = point_lotteries(dmodel, n, x[n], p0)
        D = ForwardDiff.jacobian(u -> _lottery_weights(dmodel, n, u, names, Val(n_x)), u0)   # K × (n_x + n_p)
        for r in 1:K
            (j, k, w, λ) = lot[r]
            for (t, ω, sgn) in ((_linear_index(n_m, j, k), w * (1 - λ), -w), (_linear_index(n_m, j, k + 1), w * λ, w))
                cp += 1
                Ip[cp] = n; Jp[cp] = t; Vp[cp] = ω
                for l in 1:n_x
                    cx += 1
                    Ix[cx] = t; Jx[cx] = (n - 1) * n_x + l; Vx[cx] = μ[n] * sgn * D[r, l]
                end
                for l in 1:n_p
                    Π_p[t, l] += μ[n] * sgn * D[r, n_x + l]
                end
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
    _aggregate_derivatives(model, dmodel, μ, x, y, z, Val(names), Val(length(eltype(x.data))))
end

# individual equilibrium conditions at point n, as a function of u = (x_n, p, y)
@inline function _equilibrium_u(model, dmodel, n, u::SVector, z, names::NTuple{n_p,Symbol}, yvars::NTuple{n_y,Symbol}, ::Val{n_x}) where {n_p, n_y, n_x}
    x_n = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
    p = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
    yy = NamedTuple{yvars}(ntuple(l -> u[n_x+n_p+l], Val(n_y)))
    _, s = _state(dmodel, n, p)
    equilibrium(model, s.val, x_n, yy, z, yy, z)
end

function _aggregate_derivatives(model, dmodel, μ, x, y, z, ::Val{names}, ::Val{n_x}) where {names, n_x}
    N = length(x)
    n_p = length(names)
    yvars = keys(y)
    n_y = length(yvars)
    pv = _as_vector(_current_prices(dmodel, names))
    yv = _as_vector(y)
    e(n, u) = _equilibrium_u(model, dmodel, n, u, z, names, yvars, Val(n_x))

    n_A = length(e(1, vcat(x[1], pv, yv)))
    A = zeros(n_A)
    A_μ = zeros(n_A, N)
    A_x = zeros(n_A, N * n_x)
    A_p = zeros(n_A, n_p)
    A_y = zeros(n_A, n_y)
    @inbounds for n in 1:N
        u = vcat(x[n], pv, yv)
        en = e(n, u)
        D = ForwardDiff.jacobian(v -> e(n, v), u)
        for a in 1:n_A
            A[a] += μ[n] * en[a]
            A_μ[a, n] = en[a]
            for l in 1:n_x
                A_x[a, (n-1)*n_x+l] = μ[n] * D[a, l]
            end
            for l in 1:n_p
                A_p[a, l] += μ[n] * D[a, n_x+l]
            end
            for l in 1:n_y
                A_y[a, l] += μ[n] * D[a, n_x+n_p+l]
            end
        end
    end
    return (; A, A_μ, A_x, A_p, A_y)
end

###
### Total derivative
###

# solves (I-Π')dμ = r (all equations but the last one) with Σdμ = 0
_solve_distribution(D::DistributionSolver, r::AbstractVector) = _solve_mass(D, r, 0.0)
_solve_distribution(Π::AbstractMatrix, r::AbstractVector) = _solve_distribution(_distribution_factorization(Π), r)

_flatten(x::Dolo.GArray) = collect(reinterpret(eltype(eltype(x.data)), x.data))

"""
    excess_jacobian(model, res; z=calibrated)

Jacobian `dA/dy` of the aggregated equilibrium conditions w.r.t. the aggregates,
at the point described by `res` (the output of `excess`). Computed exactly, by
the implicit function theorem.
"""
excess_jacobian(model::HModel, res; z::NamedTuple=calibrated_exogenous(model)) =
    excess_derivatives(model, res; z=z).dA

"""
    excess_derivatives(model, res; z=calibrated)

Returns `(; dA, dx, factorization)`: the jacobian `dA/dy` (see `excess_jacobian`),
the derivative of the agent's controls w.r.t. the aggregates (flattened,
`N n_x × n_y`), which gives a first order prediction of the controls at nearby
aggregates, and the solver for the agent's sparse jacobian (a `SparseSolver`,
or `nothing` when not available), which can be reused to solve the agent's
problem nearby.
"""
function excess_derivatives(model::HModel, res; z::NamedTuple=calibrated_exogenous(model))
    (; dmodel, sol, x, μ, y) = res
    φ = sol.dr
    names = price_names(model, y, z)

    p_y = projection_jacobian(model, y, z)
    (; A_μ, A_x, A_p, A_y) = aggregate_derivatives(model, dmodel, μ, x, y, z, names)

    # household: sparse jacobian if available (all derivatives in one pass on
    # the grid, see `agent_pass`), GMRES on Dolo's operators otherwise
    sparse_agent = _sparse_jacobian_available(dmodel, φ)
    Mf = nothing
    if sparse_agent
        pass = agent_pass(dmodel, x, names)
        (; Π, Π_x, Π_p) = distribution_derivatives(pass, μ)
        F_p = price_derivatives(pass, dmodel.grid)
        Mf = SparseSolver(jacobian_matrix(pass))
    else
        (; Π, Π_x, Π_p) = distribution_derivatives(dmodel, x, μ, names)
        (; J, L, F_p) = household_derivatives(dmodel, x, φ, names)
        precondition!(J, L)
    end
    # the factorization used for the ergodic distribution, if available
    Df = hasproperty(res, :Df) ? res.Df : _distribution_factorization(Π)

    n_y = length(y)
    dA = zeros(size(A_y, 1), n_y)
    dx_dy = zeros(length(x) * length(x[1]), n_y)
    for c in 1:n_y
        dp = p_y[:, c]
        b = Dolo.GArray(dmodel.grid, [-(F_p[n] * dp) for n in 1:length(x)])
        if sparse_agent
            dx = Mf \ _flatten(b)
        else
            dx = _flatten(solve_household(J, L, b))
        end
        dμ = _solve_distribution(Df, Π_x * dx + Π_p * dp)
        dA[:, c] = A_μ * dμ + A_x * dx + A_p * dp + A_y[:, c]
        dx_dy[:, c] = dx
    end
    return (; dA, dx=dx_dy, factorization=Mf)
end
