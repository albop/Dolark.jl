# One pass on the grid computing everything needed about the agent at (x, p):
#
# - residuals of the Euler equations `r = F(x, φ(x))` (as `Dolo.F`, with linear
#   interpolation of x on the 1d endogenous grid);
# - optionally, their derivatives: F_x (block diagonal), F_p (prices) and the
#   blocks of F_X (future controls, at the two nodes surrounding each future state);
# - the lotteries of the distribution (Young's method) and, optionally, the
#   derivatives of their weights w.r.t. x and p.
#
# Future states are computed once per (point, future exogenous node), and the
# derivatives come from a single evaluation with dual numbers w.r.t.
# u = (x_n, p, ΔX_1, ..., ΔX_K), where ΔX_j is a perturbation of the
# interpolated future controls. Interpolation cells are located by walking from
# the cell found at the previous point (future states are monotone in the
# endogenous state in most models), on precomputed nodes.

# nodes of a 1d endogenous grid
_nodes(g::Dolo.CGrid{1}) = (r = g.ranges[1]; collect(range(r[1], r[2]; length=r[3])))
_nodes(g::Dolo.AGrid1D) = [g[k] for k in 1:length(g)]

# cell (k, k+1) containing y (first or last cell when extrapolating), walking from k
@inline function _walk(nodes::Vector{Float64}, y::Float64, k::Int)
    n = length(nodes)
    k = max(min(k, n - 1), 1)
    @inbounds while k > 1 && y < nodes[k]
        k -= 1
    end
    @inbounds while k < n - 1 && y >= nodes[k+1]
        k += 1
    end
    return k
end

"""
Results of `agent_pass` (N grid points, K future exogenous nodes, n_x controls,
n_p prices). For each point n and node j: `cell[j,n]` is the interpolation cell
of the future state, `λ[j,n]` its (unclamped) barycentric weight, and `w[j,n]`
the transition probability. If derivatives are computed: `J[n] = ∂F_n/∂x_n`,
`Fp[n] = ∂F_n/∂p`, `G[j,n] = ∂F_n/∂X_j` (future controls at node j) and
`dλ[j,n] = ∂λ/∂(x_n, p)` (zero where the lottery weight is clamped).
"""
struct AgentPass{n_x, n_p, Tr, TJ, TP, TG, TD}
    r::Vector{Tr}
    J::Vector{TJ}
    Fp::Vector{TP}
    G::Matrix{TG}
    cell::Matrix{Int}
    λ::Matrix{Float64}
    w::Matrix{Float64}
    dλ::Matrix{TD}
    n_m::Int
    derivatives::Bool
end

"""
    agent_pass(dmodel, x, names=(); derivatives=true)

Evaluates the agent's residuals and lotteries at controls `x` (and current
prices), with derivatives w.r.t. `x`, the future controls and the prices
`names` if `derivatives` is true. See `AgentPass`.
"""
function agent_pass(dmodel::Dolo.DYModel, x::Dolo.GArray, names::Tuple=(); derivatives::Bool=true)
    _sparse_jacobian_available(dmodel) ||
        error("`agent_pass` requires a 1d endogenous grid and a Markov chain.")
    return _agent_pass(dmodel, x, Val(names), Val(length(eltype(x.data))), _n_nodes(dmodel), derivatives)
end

# controls x_n, future controls X_j, given cells (as `Dolo.F`, with linear interpolation)
@inline function _point_residual(dmodel, i, s_endo, x_n::SVector{n_x}, p::NamedTuple, ΔX, xdata, n_m, nodes, cells,
        ::Val{K}) where {n_x, K}
    model = dmodel.model
    P = dmodel.dproc.P
    Q = dmodel.dproc.Q
    m = _exo_values(_exo_names(dmodel), Q[i], p)
    T = promote_type(eltype(m), eltype(s_endo))
    s = QP((i, s_endo), SVector{length(m) + length(s_endo), T}(m..., s_endo...))
    # (terms are returned by the closure: a variable updated inside it would be boxed)
    terms = ntuple(Val(K)) do j
        M = _exo_values(_exo_names(dmodel), Q[j], p)
        S_endo = SVector(Dolo.transition(model, s.val, x_n, M)...)
        k = cells[j]
        λ = (S_endo[1] - nodes[k]) / (nodes[k+1] - nodes[k])
        @inbounds X = (1 - λ) * xdata[j+(k-1)*n_m] + λ * xdata[j+k*n_m] + ΔX[j]
        S = QP((j, S_endo), SVector(M..., S_endo...))
        (P[i, j] * Dolo.arbitrage(dmodel, s, x_n, S, X), λ)
    end
    r = sum(first, terms)
    return Dolo.complementarities(model, s, x_n, r), SVector(map(last, terms))
end

# residuals and lottery weights at point n, as a function of u = (x_n, p, ΔX_1, ..., ΔX_K)
@inline function _point_residual_u(u::SVector, dmodel, i, s_endo, xdata, n_m, nodes, cells,
        ::Val{names}, ::Val{n_x}, ::Val{K}) where {names, n_x, K}
    n_p = length(names)
    xu = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
    pu = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
    ΔX = ntuple(j -> SVector{n_x}(ntuple(l -> u[n_x+n_p+(j-1)*n_x+l], Val(n_x))), Val(K))
    F, λs = _point_residual(dmodel, i, s_endo, xu, pu, ΔX, xdata, n_m, nodes, cells, Val(K))
    return vcat(F, λs)
end

# future states' cells at point n, walking from `kc` (updated)
@inline function _point_cells!(kc, dmodel, i, s_endo, x_n, p, nodes, ::Val{K}) where K
    model = dmodel.model
    Q = dmodel.dproc.Q
    m = _exo_values(_exo_names(dmodel), Q[i], p)
    sval = SVector(m..., s_endo...)
    return ntuple(Val(K)) do j
        M = _exo_values(_exo_names(dmodel), Q[j], p)
        y = Dolo.transition(model, sval, x_n, M)[1]
        k = _walk(nodes, Float64(y), kc[i, j])
        @inbounds kc[i, j] = k
        k
    end
end

function _agent_pass(dmodel, x, ::Val{names}, ::Val{n_x}, ::Val{K}, derivatives) where {names, n_x, K}
    grid = dmodel.grid
    n_m, n_s = size(grid)
    N = n_m * n_s
    n_p = length(names)
    nodes = _nodes(_grid1d(grid.grids[2]))
    P = dmodel.dproc.P
    p0 = _current_prices(dmodel, names)
    pv = _as_vector(p0)
    xdata = x.data

    r = Vector{SVector{n_x,Float64}}(undef, N)
    J = Vector{SMatrix{n_x,n_x,Float64,n_x*n_x}}(undef, derivatives ? N : 0)
    Fp = Vector{SMatrix{n_x,n_p,Float64,n_x*n_p}}(undef, derivatives ? N : 0)
    G = Matrix{SMatrix{n_x,n_x,Float64,n_x*n_x}}(undef, K, derivatives ? N : 0)
    dλ = Matrix{SVector{n_x+n_p,Float64}}(undef, K, derivatives ? N : 0)
    cell = Matrix{Int}(undef, K, N)
    λm = Matrix{Float64}(undef, K, N)
    w = Matrix{Float64}(undef, K, N)
    kc = ones(Int, n_m, K)
    ΔX0 = ntuple(_ -> zero(SVector{n_x,Float64}), Val(K))

    @inbounds for n in 1:N
        ss = Dolo.point(grid, n)
        i = ss.loc[1]
        s_endo = _endo_part(ss.val, Val(length(variables(dmodel.model.exogenous))))
        x_n = xdata[n]
        cells = _point_cells!(kc, dmodel, i, s_endo, x_n, p0, nodes, Val(K))
        for j in 1:K
            cell[j, n] = cells[j]
            w[j, n] = P[i, j]
        end
        if derivatives
            u0 = vcat(x_n, pv, zero(SVector{K*n_x,Float64}))
            D = ForwardDiff.jacobian(
                u -> _point_residual_u(u, dmodel, i, s_endo, xdata, n_m, nodes, cells, Val(names), Val(n_x), Val(K)),
                u0)                                # (n_x + K) × (n_x + n_p + K n_x)
            F, λs = _point_residual(dmodel, i, s_endo, x_n, p0, ΔX0, xdata, n_m, nodes, cells, Val(K))
            r[n] = F
            J[n] = SMatrix{n_x,n_x}(ntuple(c -> D[(c-1)%n_x+1, (c-1)÷n_x+1], Val(n_x*n_x)))
            Fp[n] = SMatrix{n_x,n_p}(ntuple(c -> D[(c-1)%n_x+1, n_x+(c-1)÷n_x+1], Val(n_x*n_p)))
            for j in 1:K
                c0 = n_x + n_p + (j-1)*n_x
                G[j, n] = SMatrix{n_x,n_x}(ntuple(c -> D[(c-1)%n_x+1, c0+(c-1)÷n_x+1], Val(n_x*n_x)))
                λ = λs[j]
                λm[j, n] = λ
                inside = 0 < λ < 1
                dλ[j, n] = SVector{n_x+n_p}(ntuple(l -> inside ? D[n_x+j, l] : 0.0, Val(n_x+n_p)))
            end
        else
            F, λs = _point_residual(dmodel, i, s_endo, x_n, p0, ΔX0, xdata, n_m, nodes, cells, Val(K))
            r[n] = F
            for j in 1:K
                λm[j, n] = λs[j]
            end
        end
    end
    return AgentPass{n_x,n_p,eltype(r),eltype(J),eltype(Fp),eltype(G),eltype(dλ)}(r, J, Fp, G, cell, λm, w, dλ, n_m, derivatives)
end

_maxabs(p::AgentPass) = maximum(v -> maximum(abs, v), p.r)
_flatten(p::AgentPass{n_x}) where n_x = collect(reinterpret(Float64, p.r))

# Sparse matrices are built directly in CSC format, one column at a time:
# `entries!(rows, vals, c)` writes the (at most L, unsorted, possibly repeated)
# entries of column c and returns their number. (Building from triplets with
# `sparse(I, J, V)` costs more than the rest of the assembly.)
function _csc(m::Int, n::Int, L::Int, entries!::F) where F
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, n * L)
    nzval = Vector{Float64}(undef, n * L)
    rb = Vector{Int}(undef, L)
    vb = Vector{Float64}(undef, L)
    c0 = 1
    @inbounds for c in 1:n
        colptr[c] = c0
        cnt = entries!(rb, vb, c)
        # insertion sort by row
        for a in 2:cnt
            r = rb[a]; v = vb[a]; b = a - 1
            while b >= 1 && rb[b] > r
                rb[b+1] = rb[b]; vb[b+1] = vb[b]; b -= 1
            end
            rb[b+1] = r; vb[b+1] = v
        end
        # merge repeated rows
        for a in 1:cnt
            if a > 1 && rb[a] == rowval[c0-1]
                nzval[c0-1] += vb[a]
            else
                rowval[c0] = rb[a]; nzval[c0] = vb[a]; c0 += 1
            end
        end
    end
    colptr[n+1] = c0
    resize!(rowval, c0 - 1); resize!(nzval, c0 - 1)
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

"Sparse jacobian `F_x + F_X` (flattened, `N n_x × N n_x`)."
function jacobian_matrix(p::AgentPass{n_x}) where n_x
    p.derivatives || error("derivatives were not computed")
    K, N = size(p.cell)
    n_m = p.n_m
    Nx = N * n_x
    # entries (row, column, value) of row block n, in the order of the rows
    function foreach_entry(f, n)
        @inbounds begin
            Jn = p.J[n]
            for a in 1:n_x
                r = (n - 1) * n_x + a
                for b in 1:n_x
                    f(r, (n - 1) * n_x + b, Jn[a, b])
                end
                for j in 1:K
                    k = p.cell[j, n]; λ = p.λ[j, n]; Gj = p.G[j, n]
                    t1 = (_linear_index(n_m, j, k) - 1) * n_x
                    t2 = (_linear_index(n_m, j, k + 1) - 1) * n_x
                    for b in 1:n_x
                        f(r, t1 + b, (1 - λ) * Gj[a, b])
                        f(r, t2 + b, λ * Gj[a, b])
                    end
                end
            end
        end
    end
    # CSC by counting sort: rows are visited in increasing order, so the row
    # indices of each column come sorted (repeated entries are merged)
    colptr = zeros(Int, Nx + 1)
    for n in 1:N
        foreach_entry((r, c, v) -> (@inbounds colptr[c+1] += 1), n)
    end
    colptr[1] = 1
    cumsum!(colptr, colptr)
    nz = colptr[end] - 1
    rowval = Vector{Int}(undef, nz)
    nzval = Vector{Float64}(undef, nz)
    next = colptr[1:Nx]
    for n in 1:N
        foreach_entry(n) do r, c, v
            @inbounds begin
                q = next[c]
                if q > colptr[c] && rowval[q-1] == r
                    nzval[q-1] += v
                else
                    rowval[q] = r; nzval[q] = v; next[c] = q + 1
                end
            end
        end
    end
    # compact (merged entries leave gaps at the end of their column)
    w = 1
    @inbounds for c in 1:Nx
        start = colptr[c]
        colptr[c] = w
        for q in start:next[c]-1
            rowval[w] = rowval[q]; nzval[w] = nzval[q]; w += 1
        end
    end
    colptr[Nx+1] = w
    resize!(rowval, w - 1); resize!(nzval, w - 1)
    return SparseMatrixCSC(Nx, Nx, colptr, rowval, nzval)
end

# lottery weight (clamped as in `_lottery`)
@inline _clamp01(λ) = max(min(λ, one(λ)), zero(λ))

# targets and masses of the lottery from point n to node j
@inline function _lottery_targets(p::AgentPass, j, n)
    @inbounds k = p.cell[j, n]; λ = _clamp01(p.λ[j, n]); w = p.w[j, n]
    return _linear_index(p.n_m, j, k), _linear_index(p.n_m, j, k + 1), w * (1 - λ), w * λ
end

"Transition matrix of the distribution (see `transition_matrix`)."
function transition_matrix(p::AgentPass)
    K, N = size(p.cell)
    Πt = _csc(N, N, 2K, function (rows, vals, n)    # column n of Π' (row n of Π)
        for j in 1:K
            t1, t2, m1, m2 = _lottery_targets(p, j, n)
            @inbounds rows[2j-1] = t1; vals[2j-1] = m1; rows[2j] = t2; vals[2j] = m2
        end
        2K
    end)
    return copy(transpose(Πt))
end

"`Π'μ` (distribution next period), without forming `Π`."
forward(p::AgentPass, μ::AbstractVector) = forward!(zeros(size(p.cell, 2)), p, μ)

function forward!(out::AbstractVector, p::AgentPass, μ::AbstractVector)
    K, N = size(p.cell)
    fill!(out, 0.0)
    @inbounds for n in 1:N, j in 1:K
        t1, t2, m1, m2 = _lottery_targets(p, j, n)
        out[t1] += m1 * μ[n]; out[t2] += m2 * μ[n]
    end
    return out
end

# a point with (a lot of) mass in the ergodic distribution: a few iterations of μ ← Π'μ
function _mass_point(p::AgentPass; iterations=50)
    N = size(p.cell, 2)
    v = fill(1.0 / N, N)
    w = similar(v)
    for _ in 1:iterations
        forward!(w, p, v)
        v, w = w, v
    end
    return argmax(v)
end

# stationarity system (see `_distribution_matrix(Π, n0)`), from the lotteries
function _distribution_matrix(p::AgentPass, n0::Int=0)
    K, N = size(p.cell)
    nr = n0 > 0 ? n0 : N              # replaced row
    return _csc(N, N, 2K + 2, function (rows, vals, n)   # column n: -Π[n, :] + e_n
        cnt = 0
        @inbounds for j in 1:K
            t1, t2, m1, m2 = _lottery_targets(p, j, n)
            t1 != nr && (cnt += 1; rows[cnt] = t1; vals[cnt] = -m1)
            t2 != nr && (cnt += 1; rows[cnt] = t2; vals[cnt] = -m2)
        end
        n != nr && (cnt += 1; rows[cnt] = n; vals[cnt] = 1.0)
        if n0 == 0 || n == n0
            cnt += 1; rows[cnt] = nr; vals[cnt] = 1.0
        end
        cnt
    end)
end

"`(; Π, Π_x, Π_p)` as `distribution_derivatives`, from an `AgentPass` with derivatives."
function distribution_derivatives(p::AgentPass{n_x, n_p}, μ::AbstractVector; Π=true) where {n_x, n_p}
    p.derivatives || error("derivatives were not computed")
    K, N = size(p.cell)
    n_m = p.n_m
    Π_p = zeros(N, n_p)
    Π_x = _csc(N, N * n_x, 2K, function (rows, vals, c)    # column (n, l)
        n = (c - 1) ÷ n_x + 1; l = (c - 1) % n_x + 1
        @inbounds for j in 1:K
            k = p.cell[j, n]; w = p.w[j, n]; d = p.dλ[j, n]
            rows[2j-1] = _linear_index(n_m, j, k);     vals[2j-1] = -μ[n] * w * d[l]
            rows[2j]   = _linear_index(n_m, j, k + 1); vals[2j]   = μ[n] * w * d[l]
        end
        2K
    end)
    @inbounds for n in 1:N, j in 1:K
        k = p.cell[j, n]; w = p.w[j, n]; d = p.dλ[j, n]
        t1 = _linear_index(n_m, j, k); t2 = _linear_index(n_m, j, k + 1)
        for l in 1:n_p
            Π_p[t1, l] -= μ[n] * w * d[n_x + l]
            Π_p[t2, l] += μ[n] * w * d[n_x + l]
        end
    end
    return (; Π=(Π ? transition_matrix(p) : nothing), Π_x, Π_p)
end

"`F_p` as a GArray of `n_x × n_p` matrices (see `price_derivatives`)."
price_derivatives(p::AgentPass, grid) = Dolo.GArray(grid, p.Fp)
