# Aiyagari (1994): incomplete markets with idiosyncratic income risk.
#
# Pure Julia version of `models/ayiagari.yaml`: the model is a bits object and
# its equations are methods dispatched on its type.

using StaticArrays
using Dolo: GridSpace, CartesianSpace, MarkovChain, VAR1, YModel, rouwenhorst, ×, ⟂
import Dolo: transition, arbitrage, complementarities, initial_guess
using Dolark: HModel
import Dolark: projection, equilibrium

######
###### Agent: consumption-savings problem
######

# The agent takes the wage `w` and the interest rate `r` as given: they are
# exogenous variables of the agent's problem, constant over time. Their value is
# set by the aggregate model (see `projection` below). The only stochastic
# exogenous variable is the idiosyncratic efficiency `e`.

agent = let

    name = :consumption_savings

    β = 0.96
    γ = 4.0
    ρ = 0.8      # persistence of idiosyncratic efficiency
    σ = 0.1      # volatility of idiosyncratic efficiency

    # placeholder values for prices (overwritten by the aggregate model)
    w = 1.0
    r = 0.02
    e = 0.0

    y = w        # cash on hand
    c = 0.9*y    # consumption

    calibration = (; w, r, e, y, c, β, γ, ρ, σ)

    n_e = 3
    mc = rouwenhorst(n_e, ρ, σ)
    P = convert(SMatrix{n_e, n_e}, mc.P)
    Q = SVector( (SVector(w, r, e) for e in mc.V)... )

    exogenous = MarkovChain((:w, :r, :e), P, Q)

    states = GridSpace((:w, :r, :e), Q) × CartesianSpace(; y=(0.01, 100.0))
    controls = CartesianSpace(; c=(0.0, Inf))

    YModel(name, states, controls, exogenous, calibration)

end

const AgentType = typeof(agent)

function transition(model::AgentType, s::NamedTuple, x::NamedTuple, M::NamedTuple)
    y = exp(M.e)*M.w + (s.y - x.c)*(1 + M.r)
    (; y)
end

function arbitrage(model::AgentType, s::NamedTuple, x::NamedTuple, S::NamedTuple, X::NamedTuple)
    (; β, γ) = model.calibration
    eq = 1 - β*(X.c/x.c)^(-γ)*(1 + S.r)
    (eq,)
end

# borrowing constraint: c <= y
function complementarities(model::AgentType, s::NamedTuple, x::NamedTuple, Fv::SVector)
    (Fv[1] ⟂ (s.y - x.c),)
end

initial_guess(model::AgentType, s::NamedTuple) = (; c = 0.9*s.y)

######
###### Aggregate model
######

model = let

    K = 40.0    # aggregate capital (initial guess)
    z = 0.0     # aggregate productivity

    α = 0.36
    δ = 0.025
    A = 1.0
    N = 1.0     # aggregate labour

    calibration = (; K, z, α, δ, A, N)

    aggregate = CartesianSpace(; K=(1.0, 100.0))

    # aggregate productivity (not used to compute the steady state)
    ρ_z = 0.9
    σ_z = 0.01
    exogenous = VAR1((:z,), ρ_z, @SMatrix [σ_z^2;;])

    HModel(:ayiagari, agent, aggregate, exogenous, calibration)

end

# prices, from the firm's first order conditions
function projection(model::HModel{:ayiagari}, y::NamedTuple, z::NamedTuple)
    (; α, δ, A, N) = model.calibration
    (; K) = y
    r = α*A*exp(z.z)*(N/K)^(1 - α) - δ
    w = (1 - α)*A*exp(z.z)*(K/N)^α
    (; r, w)
end

# capital market clearing: aggregate savings equal capital
function equilibrium(model::HModel{:ayiagari}, s::NamedTuple, x::NamedTuple, y::NamedTuple, z::NamedTuple, Y::NamedTuple, Z::NamedTuple)
    a = s.y - x.c    # savings
    (a - y.K,)
end

@assert isbits(model)

model
