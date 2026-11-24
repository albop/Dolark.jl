# Sequence-space jacobian (Auclert, Bardóczy, Rognlie & Straub, 2021, "Using the
# sequence-space jacobian to solve and estimate heterogeneous-agent models").
#
# Works on a `Linearization` (conventions in `linearization_types.jl`, E1-E4).
#
# Truncation at horizon T (dates 0..T-1):
# - the agent's policy is at the steady state from date T on (dx_T = 0);
# - the agent sees the prices of dates 0..T-1 only (dp_t = 0 for t ≥ T);
# - the exogenous path is z_0..z_{T-1} (dz_T = 0 in the equations of date T-1);
# - unknowns: a predetermined aggregate y_i is unknown at dates 1..T (dy_{i,0} = 0,
#   dy_{i,T} enters only through AY in the equations of date T-1), the others at
#   dates 0..T-1 (dy_{i,T} = 0);
# - equations: E3 at dates 0..T-1.
#
# Household jacobian (fake-news algorithm). For a unit shock to price l at date s
# (k = s - t periods ahead):
#   dx^{(0)} = -Fx⁻¹ Fp e_l,  dx^{(1)} = -Fx⁻¹ (FX dx^{(0)} + FP e_l),
#   dx^{(k)} = -Fx⁻¹ FX dx^{(k-1)}  (k ≥ 2),
#   𝒴_k = Ax dx^{(k)},   𝒟_k = Πx dx^{(k)} + [k=0] Πp e_l + [k=1] ΠP e_l,
#   ℰ_j = Π^j Aμ',
#   F[0, s] = 𝒴_s,  F[t, s] = ℰ_{t-1}' 𝒟_s (t ≥ 1),  J[t, s] = F[t, s] + J[t-1, s-1].
# J[t, s] is the response of Aμ dμ_t + Ax dx_t (the household part of the
# equilibrium conditions at date t) to a unit change of the price at date s.
#
# Initial distribution (contract): the economy is at the steady state at date -1
# and the date-0 distribution responds to date-0 prices, dμ_0 = ΠP dp_0 (E2 at
# t = -1): e.g. cash on hand (1 + r_0) a_{-1} + w_0 e_0. This adds ℰ_t' ΠP e_l to
# J[t, 0]. With `revaluation=false`, dμ_0 = 0 instead (both coincide when ΠP = 0).

export household_jacobians, ge_jacobians, irf_ssj

# -Fx⁻¹ as a sparse matrix when Fx is block diagonal (blocks of size b), or nothing
function _block_diagonal_inverse(Fx::SparseMatrixCSC, b::Int)
    n = size(Fx, 1)
    rows = rowvals(Fx); vals = nonzeros(Fx)
    B = zeros(b, b, n ÷ b)
    for j in 1:n, k in nzrange(Fx, j)
        i = rows[k]
        (i - 1) ÷ b == (j - 1) ÷ b || return nothing
        B[(i - 1) % b + 1, (j - 1) % b + 1, (j - 1) ÷ b + 1] = vals[k]
    end
    I_ = Vector{Int}(undef, b * n); J_ = similar(I_); V = Vector{Float64}(undef, b * n)
    c = 0
    for m in 1:(n ÷ b)
        Bi = b == 1 ? fill(1 / B[1, 1, m], 1, 1) : inv(B[:, :, m])
        o = (m - 1) * b
        for q in 1:b, r in 1:b
            c += 1
            I_[c] = o + r; J_[c] = o + q; V[c] = -Bi[r, q]
        end
    end
    return sparse(I_, J_, V, n, n)
end

# Y = Aᵀ X, with Aᵀ stored as `At` (CSC): a row-wise gather, faster than the
# scatter of a CSC product.
function _gather!(Y, At::SparseMatrixCSC, X)
    rv = rowvals(At); nz = nonzeros(At)
    m = size(Y, 2)
    if m == 1
        @inbounds for i in axes(Y, 1)
            acc = 0.0
            for k in nzrange(At, i)
                acc = muladd(nz[k], X[rv[k], 1], acc)
            end
            Y[i, 1] = acc
        end
    elseif m == 2
        @inbounds for i in axes(Y, 1)
            a1 = 0.0; a2 = 0.0
            for k in nzrange(At, i)
                v = nz[k]; r = rv[k]
                a1 = muladd(v, X[r, 1], a1); a2 = muladd(v, X[r, 2], a2)
            end
            Y[i, 1] = a1; Y[i, 2] = a2
        end
    else
        fill!(Y, 0.0)
        @inbounds for i in axes(Y, 1), k in nzrange(At, i)
            v = nz[k]; r = rv[k]
            for c in 1:m
                Y[i, c] = muladd(v, X[r, c], Y[i, c])
            end
        end
    end
    return Y
end

# policy responses dx^{(k)} (columns k n_p + l of DX, k = 0..T-1) and 𝒟_k (same columns of 𝒟)
function _policy_responses!(DX, 𝒟, lin, T, n_x)
    Nx = size(DX, 1)
    n_p = size(lin.Fp, 2)
    col(k) = (k * n_p + 1):((k + 1) * n_p)
    mFxi = _block_diagonal_inverse(lin.Fx, n_x)
    if mFxi !== nothing
        # one gather per step: [dx^{(k+1)}; 𝒟_k] = [-Fx⁻¹ FX; Πx] dx^{(k)}
        Wt = sparse([mFxi * lin.FX; lin.Πx]')
        out = Matrix{Float64}(undef, size(Wt, 2), n_p)
        DX[:, col(0)] .= mFxi * lin.Fp
        G1 = mFxi * lin.FP
        for k in 0:(T - 1)
            _gather!(out, Wt, view(DX, :, col(k)))
            view(𝒟, :, col(k)) .= view(out, (Nx + 1):size(out, 1), :)
            if k < T - 1
                view(DX, :, col(k + 1)) .= view(out, 1:Nx, :)
                k == 0 && (view(DX, :, col(1)) .+= G1)
            end
        end
    else
        F = lu(lin.Fx)
        DX[:, col(0)] .= .-(F \ lin.Fp)
        for k in 1:(T - 1)
            view(DX, :, col(k)) .= .-(F \ (lin.FX * view(DX, :, col(k - 1))))
            k == 1 && (view(DX, :, col(1)) .-= F \ lin.FP)
        end
        mul!(𝒟, lin.Πx, DX)
    end
    return DX, 𝒟
end

function _expectation_vectors(Π, Aμ, nE)
    N = size(Π, 1); n_A = size(Aμ, 1)
    E = Matrix{Float64}(undef, N, n_A * nE)
    E[:, 1:n_A] .= Aμ'
    Πt = sparse(Π')
    for j in 1:(nE - 1)
        _gather!(view(E, :, (j * n_A + 1):((j + 1) * n_A)), Πt, view(E, :, ((j - 1) * n_A + 1):(j * n_A)))
    end
    return E
end

"""
    household_jacobians(lin::Linearization, T) -> Matrix{Matrix{Float64}}

Household jacobians by the fake-news algorithm: `J[a, l]` (an `n_A × n_p` array of
`T × T` matrices) is the jacobian of the household part `Aμ dμ_t + Ax dx_t` of
equilibrium condition `a` at dates `t = 0..T-1` (rows) with respect to price `l`
(`lin.price_names[l]`) at dates `s = 0..T-1` (columns), starting from the steady
state at date -1: dμ_0 = ΠP dp_0 (or dμ_0 = 0 if `revaluation=false`).
Direct price effects (`Ap`) are not included (see `ge_jacobians`).
"""
function household_jacobians(lin::Linearization, T::Int; revaluation::Bool=true)
    T ≥ 2 || throw(ArgumentError("T must be at least 2"))
    N = size(lin.Π, 1)
    Nx = size(lin.Fx, 1)
    n_x = Nx ÷ N
    n_p = size(lin.Fp, 2)
    n_A = size(lin.Aμ, 1)

    # expectation vectors ℰ_j = Π^j Aμ' (column j n_A + a), computed concurrently
    # with the policy responses when Julia runs with several threads
    nE = revaluation ? T : T - 1
    task = Threads.@spawn _expectation_vectors(lin.Π, lin.Aμ, nE)

    DX = Matrix{Float64}(undef, Nx, n_p * T)   # dx^{(k)}, column k n_p + l (0-based k)
    𝒟 = Matrix{Float64}(undef, N, n_p * T)
    _policy_responses!(DX, 𝒟, lin, T, n_x)
    col(k) = (k * n_p + 1):((k + 1) * n_p)
    𝒴 = lin.Ax * DX                        # n_A × n_p T
    view(𝒟, :, col(0)) .+= lin.Πp
    view(𝒟, :, col(1)) .+= lin.ΠP
    E = fetch(task)::Matrix{Float64}

    ℱ = view(E, :, 1:(n_A * (T - 1)))' * 𝒟  # n_A (T-1) × n_p T

    J = [Matrix{Float64}(undef, T, T) for _ in 1:n_A, _ in 1:n_p]
    @inbounds for a in 1:n_A, l in 1:n_p
        Jal = J[a, l]
        for s in 1:T
            c = (s - 1) * n_p + l
            Jal[1, s] = 𝒴[a, c]
            for t in 2:T
                Jal[t, s] = ℱ[(t - 2) * n_A + a, c]
            end
        end
        for s in 2:T, t in 2:T
            Jal[t, s] += Jal[t - 1, s - 1]
        end
    end
    if revaluation
        R = E' * lin.ΠP                     # n_A T × n_p: ℰ_t' ΠP
        for a in 1:n_A, l in 1:n_p, t in 1:T
            J[a, l][t, 1] += R[(t - 1) * n_A + a, l]
        end
    end
    return J
end

# H[rows of a, cols of i] for the effect of the path of variable i (with price
# loadings P, direct current effect D, direct effect of next-period value DY);
# `off = 1` when the unknown dates are 1..T, 0 for dates 0..T-1.
function _fill_ge_block!(H, J, Ap, P, D, DY, offs, T)
    n_A, n_p = size(J)
    n_v = size(P, 2)
    C = Ap * P .+ D
    @inbounds for a in 1:n_A, i in 1:n_v
        B = view(H, ((a - 1) * T + 1):(a * T), ((i - 1) * T + 1):(i * T))
        off = offs[i]
        # household channel through prices; date d ↦ column d - off (1-based: d+1-off)
        for l in 1:n_p
            w = P[l, i]
            w == 0 && continue
            Jal = J[a, l]
            for d in off:(T - 1), t in 1:T
                B[t, d + 1 - off] += w * Jal[t, d + 1]
            end
        end
        # direct effects: equation at date t, variable at date t and t + 1
        for t in 0:(T - 1)
            c0 = t - off
            0 ≤ c0 ≤ T - 1 && (B[t + 1, c0 + 1] += C[a, i])
            c1 = t + 1 - off
            0 ≤ c1 ≤ T - 1 && (B[t + 1, c1 + 1] += DY[a, i])
        end
    end
    return H
end

"""
    ge_jacobians(lin::Linearization, T; revaluation=true, J=household_jacobians(lin, T; revaluation)) -> (; H_U, H_Z, G)

Jacobians of the stacked equilibrium conditions (E3 at dates 0..T-1; row
`(a-1)T + t + 1`) with respect to the unknown aggregate paths (`H_U`, column
`(i-1)T + c`) and to the exogenous paths `z_0..z_{T-1}` (`H_Z`, column `(j-1)T + t + 1`),
and the general-equilibrium jacobian `G = -H_U⁻¹ H_Z` (dU = G dZ).

The unknowns of aggregate `y_i` are its values at dates 1..T if it is
predetermined (`lin.predetermined[i]`, then dy_{i,0} = 0), at dates 0..T-1
otherwise. Prices are `dp = p_y dy + p_z dz` and
`dO_t = Σ_s J[t,s] dp_s + Ap dp_t + Ay dy_t + AY dy_{t+1} + Az dz_t + AZ dz_{t+1}`.
"""
function ge_jacobians(lin::Linearization, T::Int; revaluation::Bool=true, J=household_jacobians(lin, T; revaluation))
    n_A = size(lin.Aμ, 1)
    n_y = size(lin.Ay, 2)
    n_z = size(lin.Az, 2)
    size(J, 1) == n_A && size(J[1, 1]) == (T, T) || throw(DimensionMismatch("household jacobians of the wrong size"))
    H_U = zeros(n_A * T, n_y * T)
    H_Z = zeros(n_A * T, n_z * T)
    _fill_ge_block!(H_U, J, lin.Ap, lin.p_y, lin.Ay, lin.AY, [p ? 1 : 0 for p in lin.predetermined], T)
    _fill_ge_block!(H_Z, J, lin.Ap, lin.p_z, lin.Az, lin.AZ, zeros(Int, n_z), T)
    G = lu(H_U) \ H_Z
    G .*= -1
    return (; H_U, H_Z, G)
end

"""
    irf_ssj(lin::Linearization; T=300, shock=_default_shock(lin), revaluation=true, G=nothing, J=nothing) -> IRF

Linear impulse response by the sequence-space jacobian method, to the
exogenous path `dz_t = ρ^t shock` (dates 0..T-1). A precomputed GE jacobian `G`
(from `ge_jacobians(lin, T)`) or household jacobians `J` can be given
(`revaluation` is then ignored).
"""
function irf_ssj(lin::Linearization; T::Int=300, shock=_default_shock(lin), revaluation::Bool=true, G=nothing, J=nothing)
    if G === nothing
        G = (J === nothing ? ge_jacobians(lin, T; revaluation) : ge_jacobians(lin, T; J)).G
    end
    n_y = size(lin.Ay, 2)
    n_z = size(lin.Az, 2)
    size(G) == (n_y * T, n_z * T) || throw(DimensionMismatch("G is not of size ($(n_y*T), $(n_z*T))"))
    z = Matrix{Float64}(undef, T, n_z)
    zt = collect(Float64, shock)
    for t in 1:T
        z[t, :] .= zt
        zt = lin.ρ * zt
    end
    dU = G * vec(z)
    y = zeros(T, n_y)
    for i in 1:n_y
        u = view(dU, ((i - 1) * T + 1):(i * T))
        if lin.predetermined[i]
            y[2:T, i] .= u[1:(T - 1)]
        else
            y[:, i] .= u
        end
    end
    p = y * lin.p_y' + z * lin.p_z'
    names = (; y=lin.aggregate_names, p=lin.price_names, z=lin.exogenous_names)
    return IRF(y, p, z, names)
end
