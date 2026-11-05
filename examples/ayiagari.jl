# Aiyagari (1994): incomplete markets with idiosyncratic income risk.
#
# Pure Julia version of `models/ayiagari.yaml`: the model is a bits object and
# its equations are methods dispatched on its type (through its name).
#
# `ayiagari_model(; kwargs...)` builds the model for a given calibration; the
# equations below hold for all calibrations.

using StaticArrays
using Dolo: GridSpace, CartesianSpace, MarkovChain, VAR1, YModel, rouwenhorst, ×, ⟂
import Dolo: transition, arbitrage, complementarities, initial_guess
import Dolark
using Dolark: HModel
import Dolark: projection, equilibrium

"""
    ayiagari_model(; kwargs...)

Keyword arguments (defaults in parentheses):
- agent: `β` (0.96), `γ` (4.0, inverse EIS), idiosyncratic efficiency `e`:
  `ρ` (0.8) persistence, `σ` (0.1) volatility of innovations, `n_e` (3) number
  of Rouwenhorst states; `normalize_e` (false) rescales efficiency so that its
  stationary mean is one; `y_max` (100.0) upper bound for cash on hand.
- aggregate: `K` (40.0, initial guess), `α` (0.36), `δ` (0.025), `A` (1.0), `N` (1.0).
"""
function ayiagari_model(;
        β=0.96, γ=4.0, ρ=0.8, σ=0.1, n_e=3, normalize_e=false, y_min=0.01, y_max=100.0,
        K=40.0, α=0.36, δ=0.025, A=1.0, N=1.0)

    ######
    ###### Agent: consumption-savings problem
    ######

    # The agent takes the wage `w` and the interest rate `r` as given: they are
    # exogenous variables of the agent's problem, constant over time. Their
    # value is set by the aggregate model (see `projection` below). The only
    # stochastic exogenous variable is the idiosyncratic efficiency `e`.

    agent = let

        # placeholder values for prices (overwritten by the aggregate model)
        w = 1.0
        r = 0.02
        e = 0.0

        y = w        # cash on hand
        c = 0.9*y    # consumption

        calibration = (; w, r, e, y, c, β, γ, ρ, σ)

        mc = rouwenhorst(n_e, ρ, σ)
        P = convert(SMatrix{n_e, n_e}, mc.P)
        V = collect(mc.V)
        if normalize_e
            # stationary distribution of e, then E[exp(e)] = 1
            π = (P' - I(n_e))[1:end-1, :]
            π = [π; ones(1, n_e)] \ [zeros(n_e-1); 1.0]
            V = V .- log(sum(π .* exp.(V)))
        end
        Q = SVector( (SVector(w, r, e) for e in V)... )

        exogenous = MarkovChain((:w, :r, :e), P, Q)

        states = GridSpace((:w, :r, :e), Q) × CartesianSpace(; y=(y_min, y_max))
        controls = CartesianSpace(; c=(0.0, Inf))

        YModel(:consumption_savings, states, controls, exogenous, calibration)

    end

    ######
    ###### Aggregate model
    ######

    z = 0.0     # aggregate productivity

    calibration = (; K, z, α, δ, A, N)

    aggregate = CartesianSpace(; K=(0.1*K, 10.0*K))

    # aggregate productivity (not used to compute the steady state)
    ρ_z = 0.9
    σ_z = 0.01
    exogenous = VAR1((:z,), ρ_z, @SMatrix [σ_z^2;;])

    model = HModel(:ayiagari, agent, aggregate, exogenous, calibration)
    @assert isbits(model)
    return model
end

using LinearAlgebra: I

const AgentType = YModel{:consumption_savings}

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

# the agent's problem is solved first by the endogenous grid method, then on Dolo's grid
Dolark.cold_start(model::AgentType, dmodel) = Dolark.egm(dmodel)

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

ayiagari_model()
