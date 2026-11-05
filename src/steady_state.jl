# Steady state: find aggregate values `y` such that, when agents face prices
# `projection(model, y, z)`, the integral of the equilibrium conditions against
# the ergodic distribution is zero.

function _as_namedtuple(model::HModel, y::AbstractVector)
    vars = get_aggregate(model)
    NamedTuple{vars}(Tuple(y))
end

_as_vector(y::NamedTuple) = SVector(values(y)...)

"""
    solve_agent(dmodel; x0=nothing, tol=1e-10, verbose=false)

Solves the discretized agent's problem accurately (the aggregate residuals are
differentiated, so they must be computed precisely).

Tries, in order, until the residual is below `tol`:

1. Newton from `x0`, if given (warm start);
2. Newton from `cold_start(model, dmodel)`, if the agent defines it (e.g. the
   endogenous grid method, see `egm`);
3. accelerated time iteration (each step followed by a Newton step solved by
   GMRES) from the model's initial guess, then Newton;
4. plain time iteration, then Newton (slow but robust).
"""
function solve_agent(dmodel::Dolo.DYModel; x0=nothing, factorization=nothing, tol=1e-10, verbose=false)

    converged(sol) = sol.ε < tol

    if x0 !== nothing
        sol = _newton(dmodel, x0; tol=tol, factorization=factorization, verbose=verbose)
        converged(sol) && return sol
        verbose && @warn "Newton from x0 failed (ε=$(sol.ε))."
    end

    xc = cold_start(dmodel.model, dmodel)
    if xc !== nothing
        sol = _newton(dmodel, xc; tol=tol, verbose=verbose)
        converged(sol) && return sol
        verbose && @warn "Newton from cold start failed (ε=$(sol.ε))."
    end

    for ti in (_accelerated_ti, _plain_ti)
        sol = ti(dmodel; tol=tol, verbose=verbose)
        converged(sol) && return sol
        if _isfinite(sol.dr.values)
            sol = _newton(dmodel, sol.dr.values; tol=tol, verbose=verbose)
            converged(sol) && return sol
        end
        verbose && @warn "$(ti) failed (ε=$(sol.ε))."
    end

    return sol
end

_isfinite(x::Dolo.GArray) = all(v -> all(isfinite, v), x.data)

function _newton(dmodel, x0; tol=1e-10, factorization=nothing, verbose=false)
    # direct sparse solver when possible (see `agent_jacobian.jl`), GMRES otherwise
    _sparse_jacobian_available(dmodel) &&
        return newton_sparse(dmodel, x0; tol=tol, factorization=factorization, verbose=verbose)
    wksp = Dolo.newton_workspace(dmodel; linsolve=:gmres)
    wksp.x0.data .= x0.data
    Dolo.newton(dmodel, wksp; tol_ε=tol, linsolve=:gmres, verbose=verbose)
end

function _accelerated_ti(dmodel; tol=1e-10, verbose=false)
    wksp = Dolo.time_iteration_workspace(dmodel; accelerate=true, linsolve=:gmres)
    time_iteration(dmodel, wksp; accelerate=true, linsolve=:gmres, accelerate_wait=0,
        tol_ε=tol, tol_η=0.0, verbose=verbose)
end

_plain_ti(dmodel; tol=1e-10, verbose=false) =
    time_iteration(dmodel; T=5000, tol_ε=tol, tol_η=1e-12, verbose=verbose)

function _accelerated_ti(dmodel, x0; tol=1e-10, verbose=false)
    wksp = Dolo.time_iteration_workspace(dmodel; accelerate=true, linsolve=:gmres)
    x0 === nothing || (wksp.x0.data .= x0.data)
    time_iteration(dmodel, wksp; accelerate=true, linsolve=:gmres, accelerate_wait=0,
        tol_ε=tol, tol_η=0.0, verbose=verbose)
end

"""
    excess(model, y=calibrated; z=calibrated, discretization=Dict(), x0=nothing, verbose=false)

Solves the agent's problem given aggregates `y` (and `z`), computes the ergodic
distribution and returns the aggregated equilibrium residuals together with the
intermediate objects:  `(; A, y, agent, dmodel, sol, x, μ, Π, Df)` (`Π` is the
transition matrix and `Df` the factorization used to compute `μ`).

`x0` (a `Dolo.GArray`) can be used to warm-start the agent's solution, and
`factorization` gives an initial factorization of the agent's jacobian (see
`newton_sparse`), and `distribution` the `Df` of a nearby point (its
factorization then preconditions the computation of the distribution).
"""
function excess(model::HModel, y::NamedTuple=calibrated_aggregate(model);
        z::NamedTuple=calibrated_exogenous(model),
        discretization=Dict(),
        x0=nothing,
        factorization=nothing,
        distribution=nothing,
        verbose=false,
        tol=1e-10)

    agent = agent_model(model, y, z)
    dmodel = discretize(agent, discretization)

    sol = solve_agent(dmodel; x0=x0, factorization=factorization, tol=tol, verbose=verbose)
    x = sol.dr.values

    # (`Df` is reused by `excess_jacobian`)
    if _sparse_jacobian_available(dmodel)
        pass = agent_pass(dmodel, x; derivatives=false)
        Π = transition_matrix(pass)
        Df = _distribution_factorization(pass, distribution)
    else
        Π = transition_matrix(dmodel, x)
        Df = _distribution_factorization(Π, distribution)
    end
    μ = ergodic_distribution(Df)
    A = aggregate(model, dmodel, μ, x, y, z)

    return (; A, y, agent, dmodel, sol, x, μ, Π, Df)
end

excess(model::HModel, y::AbstractVector; kwargs...) = excess(model, _as_namedtuple(model, y); kwargs...)

# controls x + (dx/dy) Δy
function _predict(x::Dolo.GArray, dx_dy::AbstractMatrix, Δy)
    n_x = length(eltype(x.data))
    T = eltype(eltype(x.data))
    dx = reinterpret(SVector{n_x,T}, dx_dy * collect(Δy))
    return Dolo.GArray(x.grid, x.data .+ dx)
end

# keep a point inside the bounds of the aggregate space
_clamp(model::HModel, y::SVector) = clamp.(y, SVector(model.aggregate.min), SVector(model.aggregate.max))

"""
    solve_steady_state(model; y0=calibrated, z=calibrated, discretization=Dict(), tol=1e-6, maxit=50, verbose=true)

Solves for the steady-state aggregates with a (damped) Newton method. The
jacobian is computed exactly, by the implicit function theorem
(`jacobian=:exact`, see `excess_jacobian`), or by finite differences
(`jacobian=:fd`). Iterates are kept within the
bounds of `model.aggregate`.

Returns `(; y, z, A, agent, dmodel, sol, x, μ, iterations)`.
"""
function solve_steady_state(model::HModel;
        y0::NamedTuple=calibrated_aggregate(model),
        z::NamedTuple=calibrated_exogenous(model),
        discretization=Dict(),
        tol=1e-6,
        maxit=50,
        h=1e-3,
        max_backsteps=10,
        jacobian=:exact,
        verbose=true)

    jacobian in (:exact, :fd) || error("Unknown jacobian method $jacobian: use :exact or :fd.")

    vars = get_aggregate(model)
    n = length(vars)

    y = _as_vector(y0)
    res = excess(model, y; z=z, discretization=discretization)
    ε = maximum(abs, res.A)
    x0 = res.x

    verbose && println("Steady state: solving for (", join(vars, ", "), ")")
    verbose && println(" it | ", join(rpad.(string.(vars), 12), " "), " | residual")

    it = 0
    while true
        verbose && println(lpad(it, 3), " | ", join((rpad(round(v; sigdigits=8), 12) for v in y), " "), " | ", ε)
        (ε < tol || it >= maxit) && break
        it += 1

        dx_dy = nothing
        fact = nothing
        if jacobian == :exact
            (; dA, dx, factorization) = excess_derivatives(model, res; z=z)
            J, dx_dy, fact = dA, dx, factorization
        else
            # forward finite differences (warm-started)
            J = zeros(n, n)
            for j in 1:n
                δ = h * max(1.0, abs(y[j]))
                yj = setindex(y, y[j] + δ, j)
                rj = excess(model, yj; z=z, discretization=discretization, x0=x0)
                J[:, j] = (rj.A - res.A) / δ
            end
        end
        dy = -(J \ Vector(res.A))

        # backtracking
        accepted = false
        for k in 0:max_backsteps
            y1 = _clamp(model, y + SVector{n}(dy) * 0.5^k)
            # warm start, with a first order prediction of the controls when available
            x1 = dx_dy === nothing ? x0 : _predict(x0, dx_dy, y1 - y)
            r1 = excess(model, y1; z=z, discretization=discretization, x0=x1, factorization=fact, distribution=res.Df)
            ε1 = maximum(abs, r1.A)
            if ε1 < ε
                y, res, ε = y1, r1, ε1
                x0 = res.x
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

    return merge((; y=_as_namedtuple(model, y), z, iterations=it), res)
end
