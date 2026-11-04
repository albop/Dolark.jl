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
differentiated numerically, so they must be computed precisely): time
iteration from the default initial guess when `x0` is not given, then Newton
iterations from that point. Falls back on time iteration if Newton fails.
"""
function solve_agent(dmodel::Dolo.DYModel; x0=nothing, tol=1e-10, verbose=false)

    if x0 === nothing
        sol = time_iteration(dmodel; verbose=verbose)
        x0 = sol.dr.values
    end

    wksp = Dolo.newton_workspace(dmodel; linsolve=:gmres)
    wksp.x0.data .= x0.data
    sol = Dolo.newton(dmodel, wksp; tol_ε=tol, linsolve=:gmres, verbose=verbose)

    if !(sol.ε < tol)
        verbose && @warn "Newton failed (ε=$(sol.ε)): falling back on time iteration."
        wksp = Dolo.time_iteration_workspace(dmodel)
        wksp.x0.data .= x0.data
        sol = time_iteration(dmodel, wksp; tol_ε=tol, tol_η=tol, T=5000, verbose=verbose)
    end

    return sol
end

"""
    excess(model, y=calibrated; z=calibrated, discretization=Dict(), x0=nothing, verbose=false)

Solves the agent's problem given aggregates `y` (and `z`), computes the ergodic
distribution and returns the aggregated equilibrium residuals together with the
intermediate objects:  `(; A, y, agent, dmodel, sol, x, μ)`.

`x0` (a `Dolo.GArray`) can be used to warm-start the agent's solution.
"""
function excess(model::HModel, y::NamedTuple=calibrated_aggregate(model);
        z::NamedTuple=calibrated_exogenous(model),
        discretization=Dict(),
        x0=nothing,
        verbose=false,
        tol=1e-10)

    agent = agent_model(model, y, z)
    dmodel = discretize(agent, discretization)

    sol = solve_agent(dmodel; x0=x0, tol=tol, verbose=verbose)
    x = sol.dr.values

    μ = ergodic_distribution(dmodel, x)
    A = aggregate(model, dmodel, μ, x, y, z)

    return (; A, y, agent, dmodel, sol, x, μ)
end

excess(model::HModel, y::AbstractVector; kwargs...) = excess(model, _as_namedtuple(model, y); kwargs...)

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

        if jacobian == :exact
            J = excess_jacobian(model, res; z=z)
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
            r1 = excess(model, y1; z=z, discretization=discretization, x0=x0)
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
