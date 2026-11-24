# First-order (linear) dynamics around the steady state: the common
# representation used by the Reiter (state-space) and SSJ (sequence-space) methods.
#
# Variables in period t (deviations from the steady state):
#   dx_t : agent's controls on the grid, flattened (N n_x; index (n-1) n_x + l
#          for grid point n, control l, grid points as in `Dolo.enum`)
#   dμ_t : distribution on the grid (N)
#   dy_t : aggregate endogenous variables (n_y)
#   dz_t : aggregate exogenous variables (n_z)
#   dp_t = p_y dy_t + p_z dz_t : prices faced by the agent (n_p), from `projection`
#
# Equations (exact first-order conditions of the discretized model):
#   (E1) agent:        Fx dx_t + FX dx_{t+1} + Fp dp_t + FP dp_{t+1} = 0
#   (E2) distribution: dμ_{t+1} = Π' dμ_t + Πx dx_t + Πp dp_t + ΠP dp_{t+1}
#   (E3) equilibrium:  Aμ dμ_t + Ax dx_t + Ap dp_t + Ay dy_t + AY dy_{t+1}
#                          + Az dz_t + AZ dz_{t+1} = 0          (n_A = n_y equations)
#   (E4) exogenous:    dz_{t+1} = ρ dz_t + ε_{t+1},  ε ~ N(0, Σ)
#
# where the equilibrium condition of the model is
#   Σ_n μ_t[n] equilibrium(model, s_n(p_t), x_t[n], y_t, z_t, y_{t+1}, z_{t+1}) = 0
# and the agent's residual at grid point n is
#   F_n(x_t[n], x_{t+1} interpolated at the future states; p_t (current states), p_{t+1} (future states)).
# Π is the transition matrix at the steady state (`Π[n, n']`: probability of
# moving from n to n', so that μ_{t+1} = Π' μ_t); Πx, Πp, ΠP are the
# derivatives of Π(x_t, p_t, p_{t+1})' μ at the steady-state distribution
# (lotteries conserve mass: the columns of Πx, Πp, ΠP sum to zero).
#
# `predetermined[i]` is true when aggregate y_i is a state variable (its value
# at t+1 is known at t, e.g. capital K with K_{t+1} = ∫ a_t dμ_t): then
# dy_{i,0} = 0 after a shock at date 0.
#
# Initial condition of impulse responses (shock at date 0, steady state until
# date -1): the agents' states at date 0 already depend on date-0 prices (e.g.
# cash on hand y_0 = e w_0 + (1 + r_0) a_{-1}), so E2 also holds between dates
# -1 and 0:
#   dμ_0 = ΠP dp_0          (dx_{-1} = 0, dp_{-1} = 0, dμ_{-1} = 0)
# where dp_0 = p_y dy_0 + p_z dz_0 may depend on non-predetermined dy_0.

"""
    Linearization

First-order approximation of a heterogeneous-agent model around its steady
state (see `src/linearization_types.jl` for the conventions). Built by
`linearize(model, steady_state)`.
"""
struct Linearization{M, D, S}
    model::M
    dmodel::D                 # discretized agent at the steady state
    steady_state::S           # output of `solve_steady_state`
    price_names::Tuple
    aggregate_names::Tuple
    exogenous_names::Tuple
    # agent (N n_x × N n_x, N n_x × n_p)
    Fx::SparseMatrixCSC{Float64,Int}
    FX::SparseMatrixCSC{Float64,Int}
    Fp::Matrix{Float64}
    FP::Matrix{Float64}
    # distribution (N × N, N × N n_x, N × n_p)
    Π::SparseMatrixCSC{Float64,Int}
    Πx::SparseMatrixCSC{Float64,Int}
    Πp::Matrix{Float64}
    ΠP::Matrix{Float64}
    # equilibrium (n_A × N, n_A × N n_x, n_A × n_p, n_A × n_y, n_A × n_z)
    Aμ::Matrix{Float64}
    Ax::Matrix{Float64}
    Ap::Matrix{Float64}
    Ay::Matrix{Float64}
    AY::Matrix{Float64}
    Az::Matrix{Float64}
    AZ::Matrix{Float64}
    # projection (n_p × n_y, n_p × n_z)
    p_y::Matrix{Float64}
    p_z::Matrix{Float64}
    # aggregate exogenous process (n_z × n_z)
    ρ::Matrix{Float64}
    Σ::Matrix{Float64}
    predetermined::Vector{Bool}
end

function Base.show(io::IO, l::Linearization)
    N = size(l.Π, 1)
    print(io, "Linearization: ", N, " grid points, ", size(l.Fx, 1) ÷ N, " control(s), prices (",
        join(l.price_names, ", "), "), aggregates (", join(l.aggregate_names, ", "),
        "; predetermined: ", join(l.aggregate_names[l.predetermined], ", "),
        "), exogenous (", join(l.exogenous_names, ", "), ")")
end

"""
    IRF

Impulse response (deviations from the steady state), rows are dates 0..T-1:
`y` (T × n_y), `p` (T × n_p, prices), `z` (T × n_z).
"""
struct IRF
    y::Matrix{Float64}
    p::Matrix{Float64}
    z::Matrix{Float64}
    names::NamedTuple
end

Base.show(io::IO, r::IRF) = print(io, "IRF: ", size(r.y, 1), " periods, aggregates (",
    join(r.names.y, ", "), "), prices (", join(r.names.p, ", "), ")")

# default shock: one standard deviation of each innovation
_default_shock(l::Linearization) = sqrt.(diag(l.Σ))
