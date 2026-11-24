# First-order perturbation by cyclic reduction (port of `dyno.solver.solve_cr`,
# from dyno.py, itself following `dynare_preprocessor_pylib`).
#
# Solves A X² + B X + C = 0 for the stable solvent X (the one whose
# eigenvalues are the n smallest roots of det(A λ² + B λ + C)), i.e. the
# solution v_t = X v_{t-1} of the model A E_t v_{t+1} + B v_t + C v_{t-1} = 0.
# Only real n×n matrices are used: no 2n×2n companion pencil (as in QZ).
#
# - Large problems (n ≥ 200) are first solved without Möbius shift: cyclic
#   reduction then preserves the row and column supports of the coefficients
#   (equations with leads, forward-looking variables, predetermined variables)
#   and all products and linear solves are restricted to them.
# - Otherwise (or if this fails), the Möbius shift x = (w - τ)/(1 - τ w) sends
#   infinite eigenvalues to 1/τ (robustness), after the condensation of static
#   variables (zero columns in A and C).
# - Newton polishing for small problems, residual check, Blanchard-Kahn check
#   from all the eigenvalues, or from spectral radii only for large problems.
#
# The coefficients are dense matrices ("densified" from the sparse jacobian of
# the model): the iterates fill in anyway.

using ArnoldiMethod: partialschur, partialeigen, LM

export solve_cr

const _CR_STABILITY = 1 + 1e-6              # stable eigenvalues: |λ| ≤ 1 + 1e-6 (as Dynare)
const _CR_STRUCTURED_MIN_SIZE = 200
const _CR_FULL_EIGENVALUES_MAX_STATES = 1000

_nzrows(M::AbstractMatrix) = findall(vec(any(!iszero, M; dims=2)))
_nzcols(M::AbstractMatrix) = findall(vec(any(!iszero, M; dims=1)))

_matrix_scale(A, B, C) = max(opnorm(A, 1), opnorm(B, 1), opnorm(C, 1), 1.0)

function _quadratic_residual(A, B, C, X)
    R = B * X + C
    r = _nzrows(A)
    isempty(r) || (R[r, :] .+= (A[r, :] * X) * X)
    return maximum(abs, R; init=0.0)
end

# n×n matrix that is zero outside rows r and columns c (dense block M), with
# empty rows and columns pruned
struct _CRBlock
    r::Vector{Int}
    c::Vector{Int}
    M::Matrix{Float64}
end

function _crblock(r::AbstractVector{Int}, c::AbstractVector{Int}, M::AbstractMatrix)
    rr, cc = _nzrows(M), _nzcols(M)
    return _CRBlock(r[rr], c[cc], M[rr, cc])
end

# Möbius shift making B2 = τ²C - τB + A best conditioned
function _select_tau(A, B, C)
    best_tau, best_rc = 0.53, -1.0
    for t in (0.31, -0.37, 0.53, -0.59, 0.71, -0.77)
        M = t * t * C - t * B + A
        F = lu(M; check=false)
        piv = abs.(diag(F.factors))
        minimum(piv) > 0 || continue
        rc = LinearAlgebra.LAPACK.gecon!('1', copy(F.factors), opnorm(M, 1))
        rc = min(rc, minimum(piv) / maximum(piv))
        if isfinite(rc) && rc > best_rc
            best_tau, best_rc = t, rc
        end
    end
    best_rc > 1e-14 || error("Cyclic reduction: singular matrix polynomial (det(A x² + B x + C) ≡ 0).")
    return best_tau
end

# spectral radius (dense eigenvalues for small matrices, Arnoldi otherwise)
function _spectral_radius(M::AbstractMatrix)
    m = size(M, 1)
    m == 0 && return 0.0
    if m > _CR_FULL_EIGENVALUES_MAX_STATES
        try
            decomp, hist = partialschur(M; nev=3, which=:LM, tol=1e-9)
            hist.converged && return maximum(abs, partialeigen(decomp)[1])
        catch
        end
    end
    return maximum(abs, eigvals(M))
end

"""
    CRSolution

Result of `solve_cr`: the solution `X`, the sorted moduli of the 2n
generalized eigenvalues (`nothing` if only spectral radii were computed), the
spectral radius of `X` (largest stable modulus) and the smallest unstable
modulus, the number of iterations, the Möbius shift `τ` and the residual.
"""
struct CRSolution
    X::Matrix{Float64}
    eigenvalues::Union{Nothing, Vector{Float64}}
    stable_radius::Float64
    unstable_min::Float64
    iterations::Int
    tau::Float64
    residual::Float64
end

function _check_bk(ρ_s, λ_u, n; evs=nothing)
    crit = _CR_STABILITY
    if ρ_s > crit
        ns = evs === nothing ? "fewer than $n" : string(count(<=(crit), evs))
        error("Blanchard-Kahn conditions fail: no stable solution ($ns stable eigenvalues for $n " *
              "variables; largest root of the stable solvent |λ| = $ρ_s).")
    elseif λ_u <= crit
        ns = evs === nothing ? "more than $n" : string(count(<=(crit), evs))
        error("Blanchard-Kahn conditions fail: indeterminacy ($ns stable eigenvalues for $n " *
              "variables; smallest unstable |λ| = $λ_u).")
    end
end

"""
    solve_cr(A, B, C; tol=1e-13, maxit=50, condense=true, tau=nothing, eigenvalues=nothing) -> CRSolution

Solves `A X² + B X + C = 0` for the stable solvent `X` by (Möbius-shifted)
cyclic reduction, and checks the Blanchard-Kahn conditions. `tau`: Möbius
shift in (-1, 1) (by default, none for n ≥ 200, then a few candidates);
`condense`: eliminate static variables (zero columns of `A` and `C`) on the
shifted path; `eigenvalues`: compute all the eigenvalues (default: when `X`
has at most 1000 nonzero columns), otherwise only spectral radii.
"""
function solve_cr(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix;
        tol=1e-13, maxit=50, condense=true, tau=nothing, eigenvalues=nothing)
    A = Matrix{Float64}(A); B = Matrix{Float64}(B); C = Matrix{Float64}(C)
    n = size(A, 1)
    tau === nothing || -1 < tau < 1 || throw(ArgumentError("Möbius shift τ must lie in (-1, 1), got $tau."))
    if tau === nothing && n >= _CR_STRUCTURED_MIN_SIZE
        try
            return _cyclic_reduction(A, B, C; tol, maxit, tau=0.0, eigenvalues)
        catch err
            (err isa _CRFailure || err isa SingularException || err isa LAPACKException) || rethrow()
        end
    end
    if condense
        res = _solve_cr_condensed(A, B, C; tol, maxit, tau, eigenvalues)
        res === nothing || return res
    end
    return _cyclic_reduction(A, B, C; tol, maxit, tau, eigenvalues)
end

# numerical failure (as opposed to a failure of the Blanchard-Kahn conditions)
struct _CRFailure <: Exception
    msg::String
end
Base.showerror(io::IO, e::_CRFailure) = print(io, "Cyclic reduction: ", e.msg)

function _cyclic_reduction(A, B, C; tol=1e-13, maxit=50, tau=nothing, eigenvalues=nothing)
    n = size(A, 1)
    scale = _matrix_scale(A, B, C)
    tau === nothing && (tau = _select_tau(A, B, C))

    # transformed polynomial B2 W² + B1 W + B0 = 0
    if tau == 0
        B0, B1, B2 = C, B, A
    else
        B0 = C - tau * B + tau^2 * A
        B1 = -2tau * C + (1 + tau^2) * B - 2tau * A
        B2 = tau^2 * C - tau * B + A
    end

    # Bk ~ B0 and Ck ~ B2 are stored as blocks on their (shrinking) supports.
    # Accumulated corrections: Ahat = B1 - Σ Ck Ak⁻¹ Bk (rows of B2 only) gives
    # W = -Ahat⁻¹ B0; Atil = B1 - Σ Bk Ak⁻¹ Ck (columns of B2 only) gives the
    # solution V = -Atil⁻¹ B2 of the reversed equation.
    Ak = copy(B1)
    Bk = _crblock(1:n, 1:n, B0)
    Ck = _crblock(1:n, 1:n, B2)
    rows_hat, cols_til = Ck.r, Ck.c
    pos_hat = zeros(Int, n); pos_hat[rows_hat] = 1:length(rows_hat)
    pos_til = zeros(Int, n); pos_til[cols_til] = 1:length(cols_til)
    hat = zeros(length(rows_hat), n)
    til = zeros(n, length(cols_til))

    stop = tol * max(maximum(abs, B0; init=0.0), maximum(abs, B1; init=0.0), 1.0)
    converged = false
    it = 0
    Fw = similar(Ak)
    for _ in 1:maxit
        it += 1
        copyto!(Fw, Ak)
        F = lu!(Fw; check=false)
        piv = abs.(diag(F.factors))
        minimum(piv; init=Inf) > 1e-14 * maximum(piv; init=0.0) || throw(_CRFailure("singular iterate."))
        rb, cb, rc, cc = Bk.r, Bk.c, Ck.r, Ck.c
        nb, nc = length(rb), length(rc)
        if nb + nc < length(cb) + length(cc)
            # left solves: [Ck; Bk] Ak⁻¹, restricted to their nonzero rows
            L = zeros(n, nc + nb)
            L[cc, 1:nc] = transpose(Ck.M)
            L[cb, nc+1:end] = transpose(Bk.M)
            ldiv!(transpose(F), L)
            Zc = transpose(view(L, :, 1:nc)); Zb = transpose(view(L, :, nc+1:nc+nb))
            CX = Zc[:, rb] * Bk.M          # Ck Ak⁻¹ Bk: rows rc, cols cb
            BY = Zb[:, rc] * Ck.M          # Bk Ak⁻¹ Ck: rows rb, cols cc
            Bn = -(Zb[:, rb] * Bk.M)
            Cn = -(Zc[:, rc] * Ck.M)
        else
            # right solves: Ak⁻¹ [Bk, Ck], restricted to their nonzero columns
            ncb = length(cb)
            R = zeros(n, ncb + length(cc))
            R[rb, 1:ncb] = Bk.M
            R[rc, ncb+1:end] = Ck.M
            ldiv!(F, R)
            CX = Ck.M * R[cc, 1:ncb]
            BY = Bk.M * R[cb, ncb+1:end]
            Bn = -(Bk.M * R[cb, 1:ncb])
            Cn = -(Ck.M * R[cc, ncb+1:end])
        end
        Ak[rc, cb] .-= CX
        Ak[rb, cc] .-= BY
        hat[pos_hat[rc], cb] .+= CX
        til[rb, pos_til[cc]] .+= BY
        Bk = _crblock(rb, cb, Bn)
        Ck = _crblock(rc, cc, Cn)
        # the corrections decay like (|λ_n| / |λ_{n+1}|)^(2^k)
        err = max(maximum(abs, CX; init=0.0), maximum(abs, BY; init=0.0))
        isfinite(err) || break
        if err < stop
            converged = true
            break
        end
    end
    converged || throw(_CRFailure("no convergence after $maxit iterations (the n-th and (n+1)-th " *
                                  "eigenvalues probably have the same modulus)."))

    Ahat = copy(B1); Ahat[rows_hat, :] .-= hat
    Atil = copy(B1); Atil[:, cols_til] .-= til

    # W = -Ahat⁻¹ B0 and V = -Atil⁻¹ B2 (only their nonzero columns)
    c0, c2 = _nzcols(B0), _nzcols(B2)
    W = zeros(n, n)
    W[:, c0] = -(lu(Ahat) \ B0[:, c0])
    V2 = -(lu(Atil) \ B2[:, c2])
    (all(isfinite, W) && all(isfinite, V2)) || throw(_CRFailure("singular limit (no eigenvalue gap?)."))

    # X = (W - τI)(I - τW)⁻¹
    X = tau == 0 ? W : Matrix(transpose(transpose(I - tau * W) \ transpose(W - tau * I)))

    res = _quadratic_residual(A, B, C, X)
    if n <= 500 && res > 1e-12 * scale
        X, res = _newton_polish(A, B, C, X, res)
    end
    (all(isfinite, X) && res <= 1e-8 * scale) || throw(_CRFailure("inaccurate solution (residual $res)."))

    # stable eigenvalues: those of X[p, p] (p: predetermined variables, nonzero
    # columns of C) and zeros; unstable ones from the eigenvalues μ of V:
    # w = 1/μ, x = (1 - τμ)/(μ - τ)
    p = _nzcols(C)
    eigenvalues === nothing && (eigenvalues = length(p) <= _CR_FULL_EIGENVALUES_MAX_STATES)
    unstable(μ) = (num = abs(1 - tau * μ); den = abs(μ - tau); den <= 1e-9 * num ? Inf : num / den)
    if !eigenvalues
        ρ_s = _spectral_radius(X[p, p])
        # with τ = 0, the smallest unstable modulus is 1/ρ(V)
        λ_u = tau == 0 ? 1 / _spectral_radius(V2[c2, :]) :
                         minimum(unstable, eigvals(V2[c2, :]); init=Inf)
        _check_bk(ρ_s, λ_u, n)
        return CRSolution(X, nothing, ρ_s, λ_u, it, tau, res)
    end
    λ_s = abs.(eigvals(X[p, p]))
    λ_unstable = [unstable(μ) for μ in [eigvals(V2[c2, :]); zeros(n - length(c2))]]
    evs = sort([λ_s; zeros(n - length(p)); λ_unstable])
    ρ_s = maximum(λ_s; init=0.0)
    λ_u = minimum(λ_unstable; init=Inf)
    _check_bk(evs[n], evs[n+1], n; evs)
    return CRSolution(X, evs, ρ_s, λ_u, it, tau, res)
end

# Newton steps on A X² + B X + C = 0: (AX + B) H + A H X = -R(X), solved by a
# complex Schur decomposition of X and column-wise back-substitution
function _newton_polish(A, B, C, X, res; maxit=3)
    for _ in 1:maxit
        R = A * X * X + B * X + C
        M = A * X + B
        S = schur(complex(X))
        T, Q = S.T, S.Z
        RQ = R * Q
        n = size(X, 2)
        H = zeros(ComplexF64, size(X))
        for j in 1:n
            rhs = -RQ[:, j] - A * (H[:, 1:j-1] * T[1:j-1, j])
            H[:, j] = (M + T[j, j] * A) \ rhs
        end
        Xn = X + real(H * Q')
        rn = _quadratic_residual(A, B, C, Xn)
        rn < res || break
        X, res = Xn, rn
    end
    return X, res
end

# Cyclic reduction after elimination of the static variables s (zero columns
# in A and C): with an orthogonal Q such that Q' B[:, s] = [R; 0], the last
# n - n_s rotated equations only involve the dynamic variables d. Returns
# `nothing` when condensation does not apply.
function _solve_cr_condensed(A, B, C; tol=1e-13, maxit=50, tau=nothing, eigenvalues=nothing)
    n = size(A, 1)
    is_dyn = vec(any(!iszero, A; dims=1)) .| vec(any(!iszero, C; dims=1))
    d = findall(is_dyn); s = findall(.!is_dyn)
    nd, ns = length(d), length(s)
    ns == 0 && return nothing
    Q = qr(B[:, s]).Q * Matrix(1.0I, n, n)    # full n×n orthogonal factor
    Qt = transpose(Q)
    QA = Qt * A[:, d]; QB = Qt * B; QC = Qt * C[:, d]
    R = QB[1:ns, s]
    cond(R) < 1e12 || return nothing
    # each static variable contributes one zero and one infinite eigenvalue
    full(evs) = evs === nothing ? nothing : sort([evs; zeros(ns); fill(Inf, ns)])
    if nd > 0
        sd = _cyclic_reduction(QA[ns+1:end, :], QB[ns+1:end, d], QC[ns+1:end, :]; tol, maxit, tau, eigenvalues)
        Xd = sd.X
    else
        sd = CRSolution(zeros(0, 0), Float64[], 0.0, Inf, 0, something(tau, 0.0), 0.0)
        Xd = zeros(0, 0)
    end
    Xs = -(R \ (QA[1:ns, :] * Xd * Xd + QB[1:ns, d] * Xd + QC[1:ns, :]))
    X = zeros(n, n)
    X[d, d] = Xd
    X[s, d] = Xs
    res = _quadratic_residual(A, B, C, X)
    (all(isfinite, X) && res <= 1e-8 * _matrix_scale(A, B, C)) || return nothing
    return CRSolution(X, full(sd.eigenvalues), sd.stable_radius, sd.unstable_min, sd.iterations, sd.tau, res)
end
