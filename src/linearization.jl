# First-order dynamics around the steady state: `linearize(model, ss)` builds a
# `Linearization` (see `linearization_types.jl` for the conventions E1-E4).
#
# The agent's blocks come from one pass on the grid (as `agent_pass`), with one
# dual-number evaluation per point w.r.t. u = (x_n, p_t, p_{t+1}, ΔX_1..ΔX_K):
# current prices enter the current state (exogenous values at node i), future
# prices the future states (exogenous values at node j, hence the transition
# and the interpolated future controls).

export linearize

# residuals and lottery weights at point n, as a function of u = (x_n, p, pf, ΔX_1, ..., ΔX_K)
@inline function _point_residual_lin(u::SVector, dmodel, i, s_endo, xdata, n_m, nodes, cells,
        ::Val{names}, ::Val{n_x}, ::Val{K}) where {names, n_x, K}
    n_p = length(names)
    xu = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
    pu = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
    pf = NamedTuple{names}(ntuple(l -> u[n_x+n_p+l], Val(n_p)))
    ΔX = ntuple(j -> SVector{n_x}(ntuple(l -> u[n_x+2n_p+(j-1)*n_x+l], Val(n_x))), Val(K))
    F, λs = _point_residual(dmodel, i, s_endo, xu, pu, pf, ΔX, xdata, n_m, nodes, cells, Val(K))
    return vcat(F, λs)
end

# An `AgentPass` w.r.t. (x, current prices), plus the derivatives w.r.t. future
# prices: `FP[n] = ∂F_n/∂p_{t+1}` and `dλP[j,n] = ∂λ/∂p_{t+1}` (zero where clamped).
function _linear_pass(dmodel, x, ::Val{names}, ::Val{n_x}, ::Val{K}) where {names, n_x, K}
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
    J = Vector{SMatrix{n_x,n_x,Float64,n_x*n_x}}(undef, N)
    Fp = Vector{SMatrix{n_x,n_p,Float64,n_x*n_p}}(undef, N)
    FP = Vector{SMatrix{n_x,n_p,Float64,n_x*n_p}}(undef, N)
    G = Matrix{SMatrix{n_x,n_x,Float64,n_x*n_x}}(undef, K, N)
    dλ = Matrix{SVector{n_x+n_p,Float64}}(undef, K, N)
    dλP = Matrix{SVector{n_p,Float64}}(undef, K, N)
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
        u0 = vcat(x_n, pv, pv, zero(SVector{K*n_x,Float64}))
        D = ForwardDiff.jacobian(
            u -> _point_residual_lin(u, dmodel, i, s_endo, xdata, n_m, nodes, cells, Val(names), Val(n_x), Val(K)),
            u0)                          # (n_x + K) × (n_x + 2 n_p + K n_x)
        F, λs = _point_residual(dmodel, i, s_endo, x_n, p0, ΔX0, xdata, n_m, nodes, cells, Val(K))
        r[n] = F
        J[n] = SMatrix{n_x,n_x}(ntuple(c -> D[(c-1)%n_x+1, (c-1)÷n_x+1], Val(n_x*n_x)))
        Fp[n] = SMatrix{n_x,n_p}(ntuple(c -> D[(c-1)%n_x+1, n_x+(c-1)÷n_x+1], Val(n_x*n_p)))
        FP[n] = SMatrix{n_x,n_p}(ntuple(c -> D[(c-1)%n_x+1, n_x+n_p+(c-1)÷n_x+1], Val(n_x*n_p)))
        for j in 1:K
            c0 = n_x + 2n_p + (j-1)*n_x
            G[j, n] = SMatrix{n_x,n_x}(ntuple(c -> D[(c-1)%n_x+1, c0+(c-1)÷n_x+1], Val(n_x*n_x)))
            λ = λs[j]
            λm[j, n] = λ
            inside = 0 < λ < 1
            dλ[j, n] = SVector{n_x+n_p}(ntuple(l -> inside ? D[n_x+j, l] : 0.0, Val(n_x+n_p)))
            dλP[j, n] = SVector{n_p}(ntuple(l -> inside ? D[n_x+j, n_x+n_p+l] : 0.0, Val(n_p)))
        end
    end
    pass = AgentPass{n_x,n_p,eltype(r),eltype(J),eltype(Fp),eltype(G),eltype(dλ)}(r, J, Fp, G, cell, λm, w, dλ, n_m, true)
    return pass, FP, dλP
end

# vector of n_x × n_p blocks -> (N n_x) × n_p matrix
function _stack_blocks(v::Vector{<:SMatrix{n_x,n_p}}) where {n_x, n_p}
    M = Matrix{Float64}(undef, length(v) * n_x, n_p)
    @inbounds for n in eachindex(v), l in 1:n_p, a in 1:n_x
        M[(n-1)*n_x+a, l] = v[n][a, l]
    end
    return M
end

# F_x: block diagonal
function _block_diagonal(p::AgentPass{n_x}) where n_x
    N = length(p.J)
    return _csc(N * n_x, N * n_x, n_x, function (rows, vals, c)
        n = (c - 1) ÷ n_x + 1; b = (c - 1) % n_x + 1
        @inbounds for a in 1:n_x
            rows[a] = (n - 1) * n_x + a; vals[a] = p.J[n][a, b]
        end
        n_x
    end)
end

# F_X: future controls at the nodes surrounding each future state (built by rows)
function _future_jacobian(p::AgentPass{n_x}) where n_x
    K, N = size(p.cell)
    n_m = p.n_m
    FXt = _csc(N * n_x, N * n_x, 2K * n_x, function (rows, vals, c)   # column c of F_X' = row c of F_X
        n = (c - 1) ÷ n_x + 1; a = (c - 1) % n_x + 1
        cnt = 0
        @inbounds for j in 1:K
            k = p.cell[j, n]; λ = p.λ[j, n]; Gj = p.G[j, n]
            t1 = (_linear_index(n_m, j, k) - 1) * n_x
            t2 = (_linear_index(n_m, j, k + 1) - 1) * n_x
            for b in 1:n_x
                cnt += 1; rows[cnt] = t1 + b; vals[cnt] = (1 - λ) * Gj[a, b]
                cnt += 1; rows[cnt] = t2 + b; vals[cnt] = λ * Gj[a, b]
            end
        end
        cnt
    end)
    return copy(transpose(FXt))
end

# derivative of Π(x, p, pf)'μ w.r.t. pf
function _future_price_distribution(p::AgentPass, dλP::Matrix{SVector{n_p,Float64}}, μ) where n_p
    K, N = size(p.cell)
    n_m = p.n_m
    ΠP = zeros(N, n_p)
    @inbounds for n in 1:N, j in 1:K
        k = p.cell[j, n]; w = p.w[j, n]; d = dλP[j, n]
        t1 = _linear_index(n_m, j, k); t2 = _linear_index(n_m, j, k + 1)
        for l in 1:n_p
            ΠP[t1, l] -= μ[n] * w * d[l]
            ΠP[t2, l] += μ[n] * w * d[l]
        end
    end
    return ΠP
end

# individual equilibrium conditions at point n, as a function of u = (x_n, p, y, z, Y, Z)
@inline function _equilibrium_lin(model, dmodel, n, u::SVector, ::Val{names}, ::Val{yv}, ::Val{zv}, ::Val{n_x}) where {names, yv, zv, n_x}
    n_p = length(names); n_y = length(yv); n_z = length(zv)
    x_n = SVector{n_x}(ntuple(l -> u[l], Val(n_x)))
    p = NamedTuple{names}(ntuple(l -> u[n_x+l], Val(n_p)))
    o = n_x + n_p
    y = NamedTuple{yv}(ntuple(l -> u[o+l], Val(n_y)))
    z = NamedTuple{zv}(ntuple(l -> u[o+n_y+l], Val(n_z)))
    Y = NamedTuple{yv}(ntuple(l -> u[o+n_y+n_z+l], Val(n_y)))
    Z = NamedTuple{zv}(ntuple(l -> u[o+2n_y+n_z+l], Val(n_z)))
    _, s = _state(dmodel, n, p)
    equilibrium(model, s.val, x_n, y, z, Y, Z)
end

function _equilibrium_derivatives(model, dmodel, μ, x, y::NamedTuple{yv}, z::NamedTuple{zv}, ::Val{names}, ::Val{n_x}) where {yv, zv, names, n_x}
    N = length(x)
    n_p = length(names); n_y = length(yv); n_z = length(zv)
    pv = _as_vector(_current_prices(dmodel, names))
    yvec = _as_vector(y); zvec = _as_vector(z)
    e(n, u) = _equilibrium_lin(model, dmodel, n, u, Val(names), Val(yv), Val(zv), Val(n_x))
    u1 = vcat(x[1], pv, yvec, zvec, yvec, zvec)
    n_A = length(e(1, u1))
    Aμ = zeros(n_A, N); Ax = zeros(n_A, N * n_x); Ap = zeros(n_A, n_p)
    Ay = zeros(n_A, n_y); AY = zeros(n_A, n_y); Az = zeros(n_A, n_z); AZ = zeros(n_A, n_z)
    o = n_x + n_p
    @inbounds for n in 1:N
        u = vcat(x[n], pv, yvec, zvec, yvec, zvec)
        en = e(n, u)
        D = ForwardDiff.jacobian(v -> e(n, v), u)
        for a in 1:n_A
            Aμ[a, n] = en[a]
            for l in 1:n_x
                Ax[a, (n-1)*n_x+l] = μ[n] * D[a, l]
            end
            for l in 1:n_p
                Ap[a, l] += μ[n] * D[a, n_x+l]
            end
            for l in 1:n_y
                Ay[a, l] += μ[n] * D[a, o+l]
                AY[a, l] += μ[n] * D[a, o+n_y+n_z+l]
            end
            for l in 1:n_z
                Az[a, l] += μ[n] * D[a, o+n_y+l]
                AZ[a, l] += μ[n] * D[a, o+2n_y+n_z+l]
            end
        end
    end
    n_A == n_y || error("The number of equilibrium conditions ($n_A) differs from the number of aggregates ($n_y).")
    return (; Aμ, Ax, Ap, Ay, AY, Az, AZ)
end

# ∂p/∂y and ∂p/∂z
function _projection_derivatives(model::HModel, y::NamedTuple{yv}, z::NamedTuple{zv}) where {yv, zv}
    p_y = ForwardDiff.jacobian(u -> projection_vector(model, NamedTuple{yv}(Tuple(u)), z), _as_vector(y))
    p_z = ForwardDiff.jacobian(u -> projection_vector(model, y, NamedTuple{zv}(Tuple(u))), _as_vector(z))
    return Matrix{Float64}(p_y), Matrix{Float64}(p_z)
end

_as_matrix(a::Number, n) = Matrix{Float64}(a * I(n))
_as_matrix(a::AbstractMatrix, n) = Matrix{Float64}(a)
_as_matrix(a::AbstractVector, n) = length(a) == n * n ? Matrix{Float64}(reshape(a, n, n)) : Matrix{Float64}(Diagonal(a))

function _exogenous_process(e::Dolo.VAR1, n_z)
    return _as_matrix(e.ρ, n_z), _as_matrix(e.Σ, n_z)
end
_exogenous_process(e, n_z) = error("Only VAR1 aggregate exogenous processes are supported (got $(typeof(e))).")

# automatic detection: y_i does not enter the date-t conditions directly but y_{i,t+1} does
function _predetermined(Ay, AY, Ap, p_y)
    Ad = Ay + Ap * p_y
    return [all(iszero, view(Ad, :, i)) && !all(iszero, view(AY, :, i)) for i in 1:size(Ay, 2)]
end

_predetermined_override(::Nothing, names, auto) = auto
function _predetermined_override(v::AbstractVector{Bool}, names, auto)
    length(v) == length(names) || error("`predetermined` must have one entry per aggregate $(names).")
    return collect(v)
end
function _predetermined_override(v::Union{Tuple,AbstractVector{Symbol}}, names, auto)
    for s in v
        s in names || error("Unknown aggregate $s (aggregates: $(names)).")
    end
    return [s in v for s in names]
end

"""
    linearize(model::HModel, ss; predetermined=nothing) -> Linearization

First-order approximation of the dynamics of `model` around the steady state
`ss` (the output of `solve_steady_state`). The agent's problem must be on a 1d
endogenous grid with a Markov chain (see `agent_pass`).

`predetermined` lists the aggregates that are state variables (a vector of
`Bool`, or a tuple of names). By default, `y_i` is predetermined when it does
not enter the equilibrium conditions at date t (directly or through prices)
but its value at t+1 does.
"""
function linearize(model::HModel, ss; predetermined=nothing)
    (; dmodel, x, μ, y, z) = ss
    _sparse_jacobian_available(dmodel) ||
        error("`linearize` requires a 1d endogenous grid and a Markov chain.")
    names = price_names(model, y, z)
    n_x = length(eltype(x.data))
    pass, FPb, dλP = _linear_pass(dmodel, x, Val(names), Val(n_x), _n_nodes(dmodel))
    N = length(x)
    n_p = length(names)

    Fx = _block_diagonal(pass)
    FX = _future_jacobian(pass)
    Fp = _stack_blocks(pass.Fp)
    FP = _stack_blocks(FPb)

    (; Π, Π_x, Π_p) = distribution_derivatives(pass, μ)
    ΠP = _future_price_distribution(pass, dλP, μ)

    (; Aμ, Ax, Ap, Ay, AY, Az, AZ) = _equilibrium_derivatives(model, dmodel, μ, x, y, z, Val(names), Val(n_x))
    p_y, p_z = _projection_derivatives(model, y, z)
    ρ, Σ = _exogenous_process(model.exogenous, length(z))

    ynames = keys(y)
    pred = _predetermined_override(predetermined, ynames, _predetermined(Ay, AY, Ap, p_y))

    return Linearization(model, dmodel, ss, names, ynames, keys(z),
        Fx, FX, Fp, FP, Π, Π_x, Π_p, ΠP, Aμ, Ax, Ap, Ay, AY, Az, AZ, p_y, p_z, ρ, Σ, pred)
end
