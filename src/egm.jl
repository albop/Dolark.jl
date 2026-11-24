# Endogenous grid method (Carroll, 2006), for agents with one endogenous state,
# following the specification of dolo.py (`src/dolo/algos/egm.py` and
# `examples/models/consumption_savings_iid.yaml` there).
#
# Besides the usual `transition` and `arbitrage` equations, the agent's model
# describes its problem in terms of post-decision states `a` (e.g. savings), with
# the following methods (all variables are `NamedTuple`s; `m`, `M` are the
# current and next exogenous variables, `s`, `S` the current and next states,
# which include the exogenous ones, as in Dolo):
#
#   poststates(model) -> Dolo.CartesianSpace
#       space of post-decision states (its bounds define the default grid)
#   half_transition(model, m, a, M) -> (; s...)
#       next endogenous states, given post-decision states:  S = g̃(m, a, M)
#   expectation(model, M, S, X) -> (; z...)
#       integrand of the expectations:                        z = E[h(M, S, X)]
#   direct_response_egm(model, m, a, z) -> (; x...)
#       controls, given post-decision states and expectations: x = τ(m, a, z)
#   reverse_state(model, m, a, x) -> (; s...)
#       current endogenous states:                            s = ã(m, a, x)
#   control_bounds(model, s) -> (lb, ub)
#       bounds on controls (`NamedTuple`s); optional, the bounds of
#       `model.controls` by default
#   half_transition_depends_on_m(model) -> Bool
#       optional (default `true`); `false` if `half_transition` does not depend
#       on `m`: the terms h(M, S, X) are then computed once per next node M
#       (instead of once per pair (m, M)), which is n_m times faster
#
# For instance, for the consumption-savings problem with cash on hand `y` and
# savings `a = y - c` (see `examples/ayiagari.jl`):
#
#   half_transition:      y' = exp(e')*w' + a*(1 + r')
#   expectation:          mr = β*(c'/c̄)^(-γ)*(1 + r')
#   direct_response_egm:  c = c̄*mr^(-1/γ)
#   reverse_state:        y = a + c
#   control_bounds:       0 <= c <= y
#
# Given the decision rule x(m', s') of the next period, on an endogenous grid
# (sᵢ, xᵢ) for each exogenous node, one iteration computes, for each node m and
# each point aₖ of the post-decision grid:
#
#   zₖ = Σ_M P(m, M) h(M, Sₖ, x(M, Sₖ)),  with Sₖ = g̃(m, aₖ, M)
#   xₖ = τ(m, aₖ, zₖ),  sₖ = ã(m, aₖ, xₖ)
#
# where x(M, S) is linearly interpolated (and extrapolated) on the endogenous
# grid, then clipped to the bounds. Clipping takes care of occasionally binding
# constraints, when the lowest post-decision state is the constrained one (e.g.
# a = 0 with a borrowing limit at zero).
#
# Restrictions: Markov chain exogenous process, one endogenous state and one
# post-decision state; the endogenous grids `sᵢ` must be increasing in `a`.
#
# A model opts in to the EGM cold start of the steady-state solvers by defining
# `Dolark.cold_start(::MyAgentType, dmodel) = Dolark.egm(dmodel)`.

"""
    cold_start(model, dmodel)

Initial guess for the controls of the discretized agent `dmodel` (a `Dolo.GArray`),
or `nothing` (default) to use accelerated time iteration from `Dolo.initial_guess`.
"""
cold_start(model, dmodel) = nothing


###
### EGM interface (to be defined by the agent's model)
###

"""
    poststates(model)

Space of post-decision states of the agent (a `Dolo.CartesianSpace`), for the
endogenous grid method. See `src/egm.jl`.
"""
function poststates end

"""
    half_transition(model, m::NamedTuple, a::NamedTuple, M::NamedTuple)

Next period's endogenous states, given current exogenous variables `m`,
post-decision states `a` and next exogenous variables `M`.
"""
function half_transition end

"""
    expectation(model, M::NamedTuple, S::NamedTuple, X::NamedTuple)

Next period's term, whose conditional expectation is passed to
`direct_response_egm`. `S` contains all next period's states (exogenous included).
"""
function expectation end

"""
    direct_response_egm(model, m::NamedTuple, a::NamedTuple, z::NamedTuple)

Controls, given current exogenous variables `m`, post-decision states `a` and
expectations `z`.
"""
function direct_response_egm end

"""
    reverse_state(model, m::NamedTuple, a::NamedTuple, x::NamedTuple)

Current endogenous states, given exogenous variables `m`, post-decision states
`a` and controls `x`.
"""
function reverse_state end

"""
    control_bounds(model, s::NamedTuple) -> (lb, ub)

Lower and upper bounds (`NamedTuple`s or `SVector`s) on the controls at state
`s`. Defaults to the bounds of `model.controls`.
"""
control_bounds(model, s::NamedTuple) = (SVector(model.controls.min), SVector(model.controls.max))

"""
    half_transition_depends_on_m(model) -> Bool

Whether `half_transition(model, m, a, M)` depends on the current exogenous
variables `m` (default `true`). Defining it as `false` speeds up `egm`.
"""
half_transition_depends_on_m(model) = true


###
### Implementation
###

# NamedTuple -> SVector (and SVectors pass through)
@inline _svec(v::NamedTuple) = SVector(values(v))
@inline _svec(v::SVector) = v

@inline _nt(::Val{names}, v) where names = NamedTuple{names}(Tuple(v))

# Problem description, with variable names as type parameters (type stability)
struct EGMProblem{M, mvars, svars, avars, xvars}
    model::M
end

EGMProblem(model) = EGMProblem{typeof(model),
    variables(model.exogenous), variables(model.states),
    variables(poststates(model)), variables(model.controls)}(model)

# all states (exogenous first) from exogenous values and endogenous states
@inline _states(::EGMProblem{<:Any,mv,sv}, m::SVector, s::SVector) where {mv,sv} =
    _nt(Val(sv), vcat(m, s))

@inline function _half_transition(p::EGMProblem{<:Any,mv,sv,av}, m, a, M) where {mv,sv,av}
    S = half_transition(p.model, _nt(Val(mv), m), _nt(Val(av), a), _nt(Val(mv), M))
    return _svec(S)
end

@inline function _expectation(p::EGMProblem{<:Any,mv,sv,av,xv}, M, S, X) where {mv,sv,av,xv}
    _svec(expectation(p.model, _nt(Val(mv), M), _states(p, M, S), _nt(Val(xv), X)))
end

@inline function _direct_response(p::EGMProblem{<:Any,mv,sv,av}, m, a, z::NamedTuple) where {mv,sv,av}
    _svec(direct_response_egm(p.model, _nt(Val(mv), m), _nt(Val(av), a), z))
end

@inline function _reverse_state(p::EGMProblem{<:Any,mv,sv,av,xv}, m, a, x) where {mv,sv,av,xv}
    _svec(reverse_state(p.model, _nt(Val(mv), m), _nt(Val(av), a), _nt(Val(xv), x)))
end

@inline function _clip(p::EGMProblem, m, s, x)
    lb, ub = control_bounds(p.model, _states(p, m, s))
    return min.(max.(x, _svec(lb)), _svec(ub))
end

# Index k such that v[k] <= s < v[k+1] (1 <= k <= n-1), walking from a
# previous index: O(1) for an increasing sequence of points s.
@inline function _walk(v, k, s)
    n = length(v)
    @inbounds while k < n - 1 && v[k+1] <= s
        k += 1
    end
    @inbounds while k > 1 && v[k] > s
        k -= 1
    end
    return k
end

# Decision rule for exogenous node j at endogenous state s (an SVector of length 1),
# interpolated on the endogenous grid (sg[:,j], xg[:,j]), then clipped.
# Returns the controls and the interpolation index (for the next call).
@inline function _egm_dr(p, sg, xg, j, m, s, k)
    col = view(sg, :, j)
    k = _walk(col, k, s[1])
    @inbounds begin
        λ = (s[1] - sg[k, j]) / (sg[k+1, j] - sg[k, j])
        x = xg[k, j] + λ * (xg[k+1, j] - xg[k, j])
    end
    return _clip(p, m, s, x), k
end

# default grid of post-decision states: log-spaced from `min` (as `agrid` in sequence-jacobian)
function _poststate_grid(space, n; pivot=0.25)
    a_min, a_max = space.min[1], space.max[1]
    isfinite(a_max) || error("Supply `a_grid`: the post-decision space has an infinite upper bound.")
    g = a_min .+ exp.(range(log(pivot), log(a_max - a_min + pivot); length=n)) .- pivot
    g[1] = a_min
    return g
end

"""
    EGMResult

Result of `egm(dmodel; details=true)`: the decision rule on the grid of `dmodel`
(`x`), the endogenous grid (`s_grid[k, i]`, `x_grid[k, i]` for post-decision state
`a_grid[k]` and exogenous node `i`), and convergence information.
"""
struct EGMResult{G, V, X}
    x::G
    a_grid::V
    s_grid::Matrix{Float64}
    x_grid::Matrix{X}
    iterations::Int
    η::Float64
    converged::Bool
end

function Base.show(io::IO, r::EGMResult)
    print(io, "EGMResult: ", r.converged ? "converged" : "not converged", " in ",
        r.iterations, " iterations (η = ", r.η, "), ", length(r.a_grid), " post-decision states")
end

"""
    egm(dmodel; a_grid=nothing, n_a=min(100, number of grid points), tol=1e-3,
        maxit=10000, details=false, verbose=false)

Solves the agent's problem (a discretized `Dolo.YModel` with a Markov chain) by
the endogenous grid method, using the EGM equations of the model (see
`src/egm.jl`). Post-decision states are discretized by `a_grid` (increasing),
by default `n_a` log-spaced points on the bounds of `poststates(model)`.

The iterations start from `Dolo.initial_guess`, and stop when the endogenous grid
and the controls change by less than `tol` (supremum norm).

Returns the controls on the grid of `dmodel` (a `Dolo.GArray`), or `nothing` if
the iterations do not converge (e.g. if the problem has no stationary solution).
With `details=true`, returns an `EGMResult` instead (also when not converged).

The defaults give a coarse solution, meant as an initial guess for Newton's
method on Dolo's grid.
"""
function egm(dmodel::Dolo.DYModel; a_grid=nothing, n_a=nothing, tol=1e-3, maxit=10000,
        details=false, verbose=false)

    model = dmodel.model
    p = EGMProblem(model)

    Q = dmodel.dproc.Q
    P = Matrix(dmodel.dproc.P)
    n_m = length(Q)

    y_min, y_max, n_y = _bounds(_grid1d(dmodel.grid.grids[2]))
    if a_grid === nothing
        n_a = n_a === nothing ? min(n_y, 100) : n_a
        a_grid = _poststate_grid(poststates(model), n_a)
    end
    @assert issorted(a_grid) "a_grid must be increasing"
    A = [SVector(a) for a in a_grid]
    n_a = length(A)

    # initial decision rule: initial guess on a uniform grid of the state space
    sg = [y for y in range(y_min, y_max; length=n_a), j in 1:n_m]
    xg = [_clip(p, Q[j], SVector(sg[k, j]), _svec(Dolo.initial_guess(model, _states(p, Q[j], SVector(sg[k, j])))))
          for k in 1:n_a, j in 1:n_m]
    sg_new = similar(sg)
    xg_new = similar(xg)

    # expectations, with their names (from one evaluation)
    S0 = _half_transition(p, Q[1], A[1], Q[1])
    X0, _ = _egm_dr(p, sg, xg, 1, Q[1], S0, 1)
    h0 = expectation(model, _nt(Val(variables(model.exogenous)), Q[1]), _states(p, Q[1], S0),
        _nt(Val(variables(model.controls)), X0))
    dep = half_transition_depends_on_m(model)
    Z = fill(zero(_svec(h0)), n_a, dep ? 1 : n_m)
    H = similar(Z, dep ? 0 : n_a, dep ? 0 : n_m)

    it, η = _egm_iterate!(p, Val(keys(h0)), dep, sg, xg, sg_new, xg_new, Z, H, A, Q, P, tol, maxit, verbose)
    converged = η < tol && all(v -> all(isfinite, v), xg)
    verbose && println("EGM: $it iterations, η = $η")

    x = (converged || details) ? _egm_on_grid(p, dmodel, sg, xg) : nothing
    details && return EGMResult(x, collect(a_grid), sg, xg, it, η, converged)
    return x
end

# expectations for node i: Z[:, c] = Σ_j P[i,j] h(M_j, S, X(M_j, S)), S = g̃(m_i, a, M_j)
function _egm_expectations!(p, Z, c, i, sg, xg, A, Q, P)
    m = Q[i]
    fill!(view(Z, :, c), zero(eltype(Z)))
    for j in axes(P, 2)
        w = P[i, j]
        iszero(w) && continue
        M = Q[j]
        k = 1
        @inbounds for l in eachindex(A)
            S = _half_transition(p, m, A[l], M)
            X, k = _egm_dr(p, sg, xg, j, M, S, k)
            Z[l, c] += w * _expectation(p, M, S, X)
        end
    end
end

# same when g̃ does not depend on m: H[:, j] = h(M_j, S, X(M_j, S)), then Z = H P'
function _egm_expectations_indep!(p, Z, H, sg, xg, A, Q, P)
    for j in eachindex(Q)
        M = Q[j]
        k = 1
        @inbounds for l in eachindex(A)
            S = _half_transition(p, M, A[l], M)   # m is not used
            X, k = _egm_dr(p, sg, xg, j, M, S, k)
            H[l, j] = _expectation(p, M, S, X)
        end
    end
    fill!(Z, zero(eltype(Z)))
    @inbounds for i in axes(P, 1), j in axes(P, 2)
        w = P[i, j]
        iszero(w) && continue
        for l in eachindex(A)
            Z[l, i] += w * H[l, j]
        end
    end
end

# EGM iterations (function barrier); the result is in (sg, xg)
function _egm_iterate!(p, hnames, dep, sg, xg, sg_new, xg_new, Z, H, A, Q, P, tol, maxit, verbose)
    n_a = length(A)
    n_m = length(Q)
    it = 0
    η = Inf
    while it < maxit
        it += 1
        dep || _egm_expectations_indep!(p, Z, H, sg, xg, A, Q, P)
        for i in 1:n_m
            m = Q[i]
            c = dep ? 1 : i
            dep && _egm_expectations!(p, Z, 1, i, sg, xg, A, Q, P)
            @inbounds for l in 1:n_a
                x = _direct_response(p, m, A[l], _nt(hnames, Z[l, c]))
                s = _reverse_state(p, m, A[l], x)
                xg_new[l, i] = x
                sg_new[l, i] = s[1]
            end
        end
        η = 0.0
        @inbounds for l in eachindex(sg)
            η = max(η, abs(sg_new[l] - sg[l]), maximum(abs, xg_new[l] - xg[l]))
        end
        sg, sg_new = sg_new, sg
        xg, xg_new = xg_new, xg
        verbose && it % 100 == 0 && println("EGM | it: $it | η: $η")
        (η < tol || !isfinite(η)) && break
    end
    # the latest iterate must be in the caller's arrays
    if isodd(it)
        copyto!(sg_new, sg)
        copyto!(xg_new, xg)
    end
    return it, η
end

# decision rule on the grid of the discretized model
function _egm_on_grid(p, dmodel, sg, xg)
    Dolo.GArray(dmodel.grid, vec([
        begin
            j = ss.loc[1]
            m = dmodel.dproc.Q[j]
            s = SVector(ss.val[end])
            k = clamp(searchsortedlast(view(sg, :, j), s[1]), 1, size(sg, 1) - 1)
            first(_egm_dr(p, sg, xg, j, m, s, k))
        end
        for ss in Dolo.enum(dmodel.grid)
    ]))
end
