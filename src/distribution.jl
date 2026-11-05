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
    _grid1d(grid.grids[2])    # only 1d endogenous grids

    return _transition_matrix(dmodel, x, _n_nodes(dmodel))
end

function _transition_matrix(dmodel, x, ::Val{K}) where K
    n_m, n_s = size(dmodel.grid)
    N = n_m * n_s
    Tf = eltype(eltype(x.data))
    I = Vector{Int}(undef, 2K * N); J = similar(I); V = Vector{Tf}(undef, 2K * N)
    c = 0
    @inbounds for n in 1:N
        for (j, k, w, λ) in point_lotteries(dmodel, n, x[n], NamedTuple())
            I[c+1] = n; J[c+1] = _linear_index(n_m, j, k);     V[c+1] = w * (1 - λ)
            I[c+2] = n; J[c+2] = _linear_index(n_m, j, k + 1); V[c+2] = w * λ
            c += 2
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

# Stationarity system: (I - Π')μ = 0 has rank N-1 (any equation can be dropped,
# they sum to zero). One equation is replaced by a normalization:
# - the equation of a point n0 with positive mass by `μ[n0] = 1`, then μ is
#   rescaled to sum to one: the matrix stays as sparse as Π, with a unit
#   diagonal (a dense row `Σμ = 1` makes sparse factorizations much slower, and
#   a zero on the diagonal breaks incomplete factorizations);
# - or, if this fails (n0 = 0), the last equation by the mass condition `Σμ = 1`.

struct DistributionSolver{S}
    A::SparseMatrixCSC{Float64,Int}   # stationarity system
    S::S                              # solver for A (see `SparseSolver`)
    n0::Int                           # normalization point (0: mass condition)
    μ::Vector{Float64}                # ergodic distribution
end

Base.size(D::DistributionSolver, args...) = size(D.A, args...)

# I - Π' with row n0 replaced by e_{n0}' (n0 > 0), or the last row by ones (n0 = 0)
function _distribution_matrix(Π::SparseMatrixCSC, n0::Int=0)
    N = size(Π, 1)
    nr = n0 > 0 ? n0 : N              # replaced row
    rows = rowvals(Π)
    vals = nonzeros(Π)
    nz = nnz(Π) + 2N
    I_ = Vector{Int}(undef, nz); J_ = Vector{Int}(undef, nz); V_ = Vector{Float64}(undef, nz)
    c = 0
    for j in 1:N                      # Π[i, j] is entry (j, i) of Π'
        j == nr && continue
        for l in nzrange(Π, j)
            c += 1
            I_[c] = j; J_[c] = rows[l]; V_[c] = -vals[l]
        end
        c += 1
        I_[c] = j; J_[c] = j; V_[c] = 1.0
    end
    for n in (n0 > 0 ? (n0:n0) : (1:N))
        c += 1
        I_[c] = nr; J_[c] = n; V_[c] = 1.0
    end
    return sparse(view(I_, 1:c), view(J_, 1:c), view(V_, 1:c), N, N)
end

# a point with (a lot of) mass in the ergodic distribution: a few iterations of μ ← Π'μ
function _mass_point(Π::SparseMatrixCSC; iterations=50)
    N = size(Π, 1)
    v = fill(1.0 / N, N)
    w = similar(v)
    for _ in 1:iterations
        mul!(w, transpose(Π), v)
        v, w = w, v
    end
    return argmax(v)
end

_unit(N, n) = (e = zeros(N); e[n] = 1.0; e)

"""
    _distribution_factorization(Π, previous=nothing; rtol=1e-12)

`DistributionSolver` for the transition matrix `Π`: the ergodic distribution
and what is needed to solve linear systems in `I - Π'` (by GMRES, to the
relative tolerance `rtol`, see `SparseSolver`). The normalization point is the
mode of the distribution of `previous` (the solver at a nearby point) if
given, or found by a few iterations of `μ ← Π'μ`. `guess` is an initial guess
for the ergodic distribution.
"""
function _distribution_factorization(Π, previous=nothing; rtol=1e-12, guess=nothing)
    Π isa AgentPass || (Π = sparse(Π))     # (the matrix is built from the lotteries of an `AgentPass`)
    N = Π isa AgentPass ? size(Π.cell, 2) : size(Π, 1)
    valid(μ, n0) = all(isfinite, μ) && minimum(μ) > -1e-10 && μ[n0] > 1e-8

    candidates = previous isa DistributionSolver && size(previous, 1) == N ?
        (argmax(previous.μ), 0) : (0,)
    for n0 in candidates
        n0 == 0 && (n0 = _mass_point(Π))
        A = _distribution_matrix(Π, n0)
        S = SparseSolver(A; rtol=rtol)
        # GMRES only: if it fails, the normalization point is probably bad (little mass)
        x0 = guess === nothing || guess[n0] <= 0 ? nothing : guess ./ guess[n0]
        μ = try_solve(S, _unit(N, n0); x0=x0)
        μ === nothing && continue
        μ ./= sum(μ)
        valid(μ, n0) && return DistributionSolver(A, S, n0, μ)
    end
    # fallback: mass condition
    A = _distribution_matrix(Π, 0)
    S = SparseSolver(A)
    return DistributionSolver(A, S, 0, S \ _unit(N, N))
end

"""
Solves `(I-Π')d = r` (all equations but the last one) with `Σd = m`.
"""
function _solve_mass(D::DistributionSolver, r::AbstractVector, m::Real)
    N = length(r)
    if D.n0 > 0
        # the last equation is implied (equations sum to zero): equation n0 can be dropped
        # instead; with the normalization d[n0] = 0, the solutions are d + tμ
        b = copy(r)
        b[N] = -sum(view(r, 1:N-1))
        b[D.n0] = 0.0
        d = D.S \ b
        d .+= (m - sum(d)) .* D.μ
        return d
    else
        return D.S \ vcat(r[1:end-1], m)
    end
end

# μ'(Π - I) = 0 with Σμ = 1, from the factorization of the stationarity
# system (see `_distribution_factorization`)
ergodic_distribution(D::DistributionSolver) = D.μ

ergodic_distribution(Π::AbstractMatrix) = ergodic_distribution(_distribution_factorization(Π))

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
