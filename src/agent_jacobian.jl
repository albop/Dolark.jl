# Sparse jacobian of the agent's residuals, and Newton's method with a direct solver.
#
# With linear interpolation of the decision rule on a 1d endogenous grid, the
# residual at grid point n, F(x_n, φ(x)), depends on x_n and on the values of x
# at the two nodes surrounding each of the K future states. The jacobian
# `F_x + F_X` of `x ↦ F(x, φ(x))` is then a sparse matrix with at most
# `1 + 2K` blocks per row. It is assembled explicitly and factorized (sparse LU)
# instead of being inverted by GMRES on the linear operator (with β(1+r) close
# to one, GMRES needs many iterations, each one costing a pass on the grid).

# Interpolation cell of `y` on a 1d grid: `(k, λ)` such that the linear
# interpolant is `(1-λ) v_k + λ v_{k+1}` (λ outside of [0,1] when extrapolating),
# as in Dolo's linear splines.
@inline function _cell(g::Dolo.CGrid{1}, y)
    a, b, ng = g.ranges[1]
    u = (y - a) / (b - a) * (ng - 1)
    k = max(min(floor(Int, u), ng - 2), 0)
    return k + 1, u - k
end
@inline _cell(g::Dolo.AGrid1D, y) = Dolo.locate_cell(g, y)

# is the sparse jacobian available for this grid (and decision rule)?
_sparse_jacobian_available(dmodel) = false
_sparse_jacobian_available(dmodel::Dolo.DYModel{<:Any, <:Dolo.PGrid{<:Dolo.SGrid, <:Union{Dolo.CGrid{1}, Dolo.CartesianGrid{1}}}}) = true
_sparse_jacobian_available(dmodel, φ) = false
_sparse_jacobian_available(dmodel, φ::Dolo.DFun) =
    _sparse_jacobian_available(dmodel) && all(itp -> Dolo.spline_order(itp) == 1, φ.itp)

_maxabs(x::Dolo.GArray) = maximum(v -> maximum(abs, v), x.data)

"""
    agent_jacobian(dmodel, x, φ)

Returns `(r, M)`: the residuals `r = F(x, φ)` (a `Dolo.GArray`) and the sparse
jacobian `M = F_x + F_X` of `x ↦ F(x, φ(x))` (flattened, `N n_x × N n_x`),
where `φ` interpolates `x` linearly on a 1d endogenous grid.
"""
function agent_jacobian(dmodel::Dolo.DYModel, x::Dolo.GArray, φ)
    _sparse_jacobian_available(dmodel, φ) ||
        error("The sparse jacobian requires a 1d endogenous grid and linear interpolation.")
    pass = agent_pass(dmodel, x)
    return Dolo.GArray(x.grid, pass.r), jacobian_matrix(pass)
end

"""
    newton_sparse(dmodel, x0; tol=1e-10, maxit=50, max_backsteps=10, factorization=nothing, θ=0.25, verbose=false)

Solves `F(x, φ(x)) = 0` for the agent from `x0`, by a Newton method where
steps are solved with a `SparseSolver` of `agent_jacobian` (GMRES preconditioned
by an incomplete LU factorization).

The jacobian (and its solver) is reused across iterations (chord method) as
long as the residual decreases by a factor at least `θ` at each step; it is
recomputed otherwise. Steps made with a fresh jacobian are halved until the
residual decreases. An initial solver (e.g. for the jacobian at a nearby
solution) can be given with `factorization`.

Returns `(; dr, ε, iterations, factorizations, factorization)` where `dr`
interpolates the solution, `factorizations` is the number of jacobians computed
and `factorization` is the last solver used.
"""
function newton_sparse(dmodel::Dolo.DYModel, x0::Dolo.GArray; tol=1e-10, maxit=50, max_backsteps=10,
        factorization=nothing, θ=0.25, verbose=false)
    x = Dolo.GArray(dmodel.grid, copy(x0.data))
    φ = _decision_rule(dmodel, x)
    n_x = length(eltype(x.data))
    T = eltype(eltype(x.data))

    residual() = try
        r = Dolo.GArray(x.grid, agent_pass(dmodel, x; derivatives=false).r)
        r, _maxabs(r)
    catch err
        err isa DomainError || rethrow()
        nothing, T(Inf)    # e.g. negative consumption
    end

    r, ε = residual()
    verbose && println("Newton (sparse) | it 0 | ε = $ε")
    fact = factorization
    it = 0
    n_fact = 0
    x_old = copy(x.data)
    while ε >= tol && it < maxit && isfinite(ε)
        it += 1
        fresh = fact === nothing
        if fresh
            r, M = agent_jacobian(dmodel, x, φ)
            fact = SparseSolver(M; rtol=1e-8)     # inexact Newton steps
            n_fact += 1
        end
        dx = reinterpret(SVector{n_x,T}, fact \ _flatten(r))
        copyto!(x_old, x.data)
        accepted = false
        # with a fresh jacobian: backtracking; otherwise only the full step is tried
        for k in 0:(fresh ? max_backsteps : 0)
            x.data .= x_old .- dx .* T(0.5)^k
            Dolo.fit!(φ, x)
            r1, ε1 = residual()
            if ε1 < ε
                # slow convergence: new factorization at the next iteration
                ε1 > θ * ε && (fact = nothing)
                r, ε = r1, ε1
                accepted = true
                break
            end
        end
        if !accepted
            x.data .= x_old
            Dolo.fit!(φ, x)
            fresh && break       # no descent with the exact jacobian: give up
            fact = nothing       # stale factorization: retry with a fresh one
            r, ε = residual()
        end
        verbose && println("Newton (sparse) | it $it | ε = $ε", fresh ? " (new factorization)" : "")
    end
    return (; dr=φ, ε, iterations=it, factorizations=n_fact, factorization=fact)
end
