# Distribution of agents on the grid of a discretized agent model.
#
# The distribution lives on the same grid as the decision rule: an exogenous
# Markov chain (index i) times a 1d endogenous grid (index k). Next-period
# endogenous states falling between grid points are split between the two
# neighbouring points ("lottery", a.k.a. Young's method).

# linear index on a product grid (exogenous index varies fastest, as in `Dolo.enum`)
@inline _linear_index(n_m::Int, i::Int, k::Int) = i + (k - 1) * n_m

"""
    transition_matrix(dmodel, x)

Sparse matrix `Π` such that `Π[n,n']` is the probability of moving from grid
point `n` to grid point `n'`, given the controls `x` (a `Dolo.GArray`).
"""
function transition_matrix(dmodel::Dolo.DYModel, x::Dolo.GArray)

    grid = dmodel.grid
    endo_grid = grid.grids[2]
    endo_grid isa Dolo.CGrid{1} || error("Only 1d endogenous grids are supported for now.")

    n_m, n_s = size(grid)
    N = n_m * n_s
    Tf = eltype(eltype(x.data))

    I = Int[]
    J = Int[]
    V = Tf[]

    for n in 1:N
        for (j, k, w) in point_transitions(dmodel, n, x[n], NamedTuple())
            push!(I, n)
            push!(J, _linear_index(n_m, j, k))
            push!(V, w)
        end
    end

    return sparse(I, J, V, N, N)
end

"""
    ergodic_distribution(dmodel, x)

Stationary distribution of agents (a vector summing to one, on the grid of
`dmodel`), when they follow decisions `x`.
"""
function ergodic_distribution(dmodel::Dolo.DYModel, x::Dolo.GArray)
    Π = transition_matrix(dmodel, x)
    return ergodic_distribution(Π)
end

function ergodic_distribution(Π::AbstractMatrix)
    N = size(Π, 1)
    # solve μ'(Π - I) = 0 with Σμ = 1 (replacing the last equation)
    A = sparse(Π' - I)
    A = [A[1:end-1, :]; ones(1, N)]
    b = zeros(N)
    b[end] = 1.0
    μ = A \ b
    return μ
end

"""
    aggregate(model, dmodel, μ, x, y, z)

Integral of the equilibrium conditions against the distribution `μ`, where the
agents follow decisions `x`, at steady-state aggregates (`y`, `z`).
"""
function aggregate(model::HModel, dmodel::Dolo.DYModel, μ::AbstractVector, x::Dolo.GArray, y::NamedTuple, z::NamedTuple)
    return sum(
        μ[n] * equilibrium(model, ss.val, a, y, z, y, z)
        for (n, (ss, a)) in enumerate(zip(Dolo.enum(dmodel.grid), x))
    )
end
