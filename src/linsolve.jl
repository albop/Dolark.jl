# Sparse linear systems: the agent's jacobian and the stationarity system of the
# distribution (N × N, N = number of grid points, with O(1 + 2K) entries per row).
#
# A sparse LU factorization (UMFPACK) of these matrices costs ~10ms (N = 3500),
# more than everything else in an evaluation. Instead, systems are solved by
# GMRES, preconditioned by an incomplete LU factorization (threshold `τ`):
# about 1ms to build, 10 to 25 GMRES iterations. The exact factorization is
# only computed if GMRES fails.

import IncompleteLU

"""
    solve_preconditioned(A, P, b; rtol=1e-12, krylov_dim=60)

Solves `A x = b` by GMRES, right-preconditioned by `P` (any object supporting
`ldiv!(y, P, x)`: an incomplete or exact factorization of `A`, or of a matrix
close to `A`). Returns `x`, or `nothing` if GMRES does not converge within
`krylov_dim` iterations (relative tolerance `rtol` on the residual).
"""
function solve_preconditioned(A::AbstractMatrix, P, b::AbstractVector; rtol=1e-12, krylov_dim=60,
        workspace=Dolo.gmres_workspace(eltype(b), length(b), krylov_dim), aux=similar(b), x0=nothing)
    n = length(b)
    T = eltype(b)
    tol = rtol * norm(b)
    # with an initial guess x0: solve for the correction A δ = b - A x0
    r = x0 === nothing ? b : b - A * x0
    x0 !== nothing && norm(r) <= tol && return copy(x0)
    # (A P⁻¹) w = r,  δ = P⁻¹ w
    op! = (out, v) -> (ldiv!(aux, P, v); mul!(out, A, aux))
    w = zeros(T, n)
    _, converged = Dolo.gmres!(w, op!, r, workspace; tol=tol)
    (converged && all(isfinite, w)) || return nothing
    x = zeros(T, n)
    ldiv!(x, P, w)
    x0 === nothing || (x .+= x0)
    return x
end

"""
    SparseSolver(A; τ=1e-2, rtol=1e-12)

Solver for `A x = b` (`S \\ b`): GMRES preconditioned by the incomplete LU
factorization of `A` with drop tolerance `τ`, to a relative tolerance `rtol`
on the residual (see `solve_preconditioned`), falling back on a sparse LU
factorization (kept for later solves) if GMRES fails. Newton steps only need
`rtol` well below the rate of convergence (inexact Newton).
"""
mutable struct SparseSolver{M<:SparseMatrixCSC, P, W}
    A::M
    P::P                     # preconditioner (or nothing if the ILU failed)
    F::Union{Nothing, Factorization}
    workspace::W             # GMRES workspace (allocated once)
    aux::Vector{Float64}
    rtol::Float64
    fallbacks::Int           # number of solves where GMRES failed
end

const KRYLOV_DIM = 60

# GMRES workspaces are shared by the solvers of a task (the Krylov basis is a
# dense N × (KRYLOV_DIM+1) matrix: allocating it for each solver costs more than
# the solves)
const _GMRESWorkspace = typeof(Dolo.gmres_workspace(Float64, 1, 1))
function _gmres_workspace(n::Int)
    cache = get!(() -> Dict{Int,_GMRESWorkspace}(), task_local_storage(), :dolark_gmres)::Dict{Int,_GMRESWorkspace}
    return get!(() -> Dolo.gmres_workspace(Float64, n, KRYLOV_DIM), cache, n)
end

function SparseSolver(A::SparseMatrixCSC; τ=1e-2, rtol=1e-12)
    P = try
        IncompleteLU.ilu(A; τ=τ)
    catch err
        err isa Union{SingularException, ArgumentError} || rethrow()
        nothing
    end
    n = size(A, 1)
    return SparseSolver(A, P, nothing, _gmres_workspace(n), zeros(n), Float64(rtol), 0)
end

Base.size(S::SparseSolver, args...) = size(S.A, args...)

# GMRES only (from the initial guess x0, if given): `nothing` if it fails
function try_solve(S::SparseSolver, b::AbstractVector; x0=nothing)
    S.F === nothing || return S.F \ b
    S.P === nothing && return nothing
    return solve_preconditioned(S.A, S.P, b; rtol=S.rtol, krylov_dim=KRYLOV_DIM, workspace=S.workspace, aux=S.aux, x0=x0)
end

function Base.:\(S::SparseSolver, b::AbstractVector)
    x = try_solve(S, b)
    x === nothing || return x
    S.fallbacks += 1
    S.F = lu(S.A)
    return S.F \ b
end

Base.:\(S::SparseSolver, B::AbstractMatrix) = reduce(hcat, [S \ B[:, c] for c in 1:size(B, 2)])
