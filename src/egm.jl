# Endogenous grid method (Carroll, 2006) for the canonical consumption-savings
# agent, used to get a good initial guess for the agent's decision rule.
#
# Requirements on the agent model (`Dolo.YModel` with a Markov chain):
# - exogenous variables include `w`, `r`, `e`;
# - the endogenous state is cash on hand `y`, the control is consumption `c`;
# - `y' = exp(e')*w' + (y - c)*(1 + r')`,  Euler: `1 = β E[(c'/c)^(-γ) (1 + r')]`,
#   borrowing constraint `c <= y`;
# - `β` and `γ` are in the calibration.
#
# A model opts in by defining `Dolark.cold_start(::MyAgentType, dmodel) = Dolark.egm(dmodel)`.

"""
    cold_start(model, dmodel)

Initial guess for the controls of the discretized agent `dmodel` (a `Dolo.GArray`),
or `nothing` (default) to use accelerated time iteration from `Dolo.initial_guess`.
"""
cold_start(model, dmodel) = nothing

# consumption policy from the endogenous grid (Y[:,j], C[:,j]); below Y[1,j]
# the constraint binds; linear extrapolation above the last point
@inline function _egm_policy(Y, C, j, y)
    n = size(Y, 1)
    y <= Y[1, j] && return y
    k = searchsortedlast(view(Y, :, j), y)
    k = min(k, n - 1)
    λ = (y - Y[k, j]) / (Y[k+1, j] - Y[k, j])
    return C[k, j] + λ * (C[k+1, j] - C[k, j])
end

# x^e, computed as exp(e log x): faster than `^` (which is correctly rounded),
# accurate enough for an initial guess
@inline _pow(x, e) = exp(e * log(x))

# u[m] = R (c(y'_m))^(-γ) for an increasing sequence of points y'_m = R a_m + W
# (pointer walk on the endogenous grid (Y[:,j], C[:,j]))
function _egm_marginal_utility!(u, Y, C, j, agrid, R, W, γ)
    n = size(Y, 1)
    k = 1
    @inbounds for m in eachindex(agrid)
        y = R * agrid[m] + W
        if y <= Y[1, j]
            c = y
        else
            while k < n - 1 && Y[k+1, j] <= y
                k += 1
            end
            λ = (y - Y[k, j]) / (Y[k+1, j] - Y[k, j])
            c = C[k, j] + λ * (C[k+1, j] - C[k, j])
        end
        u[m] = R * _pow(c, -γ)
    end
    return u
end

"""
    egm(dmodel; n_a=min(100, number of grid points), tol=1e-3, maxit=10000, verbose=false)

Solves the consumption-savings problem by the endogenous grid method, on a
log-spaced grid of `n_a` savings levels in `[0, y_max]`, and returns consumption
on the grid of `dmodel` (a `Dolo.GArray`), or `nothing` if it does not
converge. See the requirements in `src/egm.jl`.

The defaults give a coarse solution, meant as an initial guess for Newton's
method on Dolo's grid (`tol` is on the change of consumption between two
iterations).
"""
function egm(dmodel::Dolo.DYModel; n_a=nothing, tol=1e-3, maxit=10000, verbose=false)

    model = dmodel.model
    (; β, γ) = model.calibration
    vars = variables(model.exogenous)
    iw = findfirst(==(:w), vars)
    ir = findfirst(==(:r), vars)
    ie = findfirst(==(:e), vars)
    Q = dmodel.dproc.Q
    P = Matrix(dmodel.dproc.P)
    n_m = length(Q)
    W = [Q[j][iw] * exp(Q[j][ie]) for j in 1:n_m]   # labour income
    R = [1 + Q[j][ir] for j in 1:n_m]               # gross return

    y_min, y_max, n_y = _bounds(_grid1d(dmodel.grid.grids[2]))
    n_a = n_a === nothing ? min(n_y, 100) : n_a
    # log-spaced savings grid on [0, y_max] (as sequence-jacobian's `agrid`)
    pivot = 0.25
    agrid = exp.(range(log(pivot), log(y_max + pivot); length=n_a)) .- pivot
    agrid[1] = 0.0

    # initial policy: c = y on a grid starting at the lowest income
    Y = [agrid[k] + W[j] for k in 1:n_a, j in 1:n_m]
    C = copy(Y)
    Cn = similar(C)         # previous iterate (convergence check)
    up = zeros(n_a, n_m)    # (1+r_j) c(j, y'_jk)^(-γ), for next state j
    mu = zeros(n_a, n_m)    # E_i[(1+r') c'^(-γ)], for current state i

    it = 0
    η = Inf
    e = -1 / γ
    while it < maxit
        it += 1
        for j in 1:n_m
            _egm_marginal_utility!(view(up, :, j), Y, C, j, agrid, R[j], W[j], γ)
        end
        mul!(mu, up, P')
        # convergence is checked every 10 iterations (as in sequence-jacobian)
        check = it % 10 == 0
        check && copyto!(Cn, C)
        @inbounds for i in 1:n_m, k in 1:n_a
            c = _pow(β * mu[k, i], e)
            C[k, i] = c
            Y[k, i] = agrid[k] + c
        end
        if check
            η = 0.0
            @inbounds for l in eachindex(C)
                η = max(η, abs(C[l] - Cn[l]))
            end
            η < tol && break
        end
    end
    verbose && println("EGM: $it iterations, |Δc| = $η")
    # no stationary solution (e.g. β(1+r) ≥ 1): let the caller use another method
    (η < tol && all(isfinite, C)) || return nothing

    return Dolo.GArray(dmodel.grid, vec([
        SVector(_egm_policy(Y, C, ss.loc[1], ss.val[end])) for ss in Dolo.enum(dmodel.grid)
    ]))
end
