# Heterogeneous-agent model, represented in pure Julia (as a bits object),
# in the same spirit as Dolo's `YModel`.
#
# An `HModel` is made of:
# - `agent`: a Dolo model (`Dolo.YModel`) describing the problem of one agent.
#    Some of its exogenous variables (e.g. prices `w`, `r`) are set by the aggregate model.
# - `aggregate`: a `Dolo.CartesianSpace` of aggregate endogenous variables `y`
#    (e.g. capital `K`), with bounds used by the solvers.
# - `exogenous`: the process for the aggregate exogenous variables `z` (e.g. TFP).
# - `calibration`: a `NamedTuple` with the values of `y`, `z` and of the parameters.
#
# The equations are defined as methods dispatched on the type of the model
# (the name is a type parameter, so one can dispatch on `HModel{:name}`):
#
#   projection(model, y::NamedTuple, z::NamedTuple) -> NamedTuple
#       values of the agent's exogenous variables (prices) implied by aggregates
#
#   equilibrium(model, s::NamedTuple, x::NamedTuple, y::NamedTuple, z::NamedTuple,
#                Y::NamedTuple, Z::NamedTuple) -> Tuple
#       individual contribution to the equilibrium conditions;
#       in equilibrium, its integral against the distribution of agents is zero.

struct HModel{N, A, Y, E, C}
    agent::A          # Dolo model of the individual agent
    aggregate::Y      # space of aggregate endogenous variables
    exogenous::E      # aggregate exogenous process
    calibration::C    # NamedTuple
end

HModel(name::Symbol, agent, aggregate, exogenous, calibration) =
    HModel{name, typeof(agent), typeof(aggregate), typeof(exogenous), typeof(calibration)}(
        agent, aggregate, exogenous, calibration
    )

name(::HModel{N}) where N = N

get_aggregate(model::HModel) = variables(model.aggregate)
get_exogenous(model::HModel) = variables(model.exogenous)

Base.eltype(model::HModel) = eltype(model.calibration)

# values from the calibration, as a NamedTuple
calibrated(model::HModel, vars::Tuple) = NamedTuple{vars}(map(v -> model.calibration[v], vars))
calibrated_aggregate(model::HModel) = calibrated(model, get_aggregate(model))
calibrated_exogenous(model::HModel) = calibrated(model, get_exogenous(model))

function recalibrate(model::HModel; kwargs...)
    calib = merge(model.calibration, kwargs)
    HModel(name(model), model.agent, model.aggregate, model.exogenous, calib)
end

"""
    projection(model::HModel, y::NamedTuple, z::NamedTuple)

Values taken by (some of) the agent's exogenous variables, given aggregate
endogenous variables `y` and aggregate exogenous variables `z`. To be defined
for each model.
"""
function projection end

"""
    equilibrium(model::HModel, s, x, y, z, Y, Z)

Individual contribution to the equilibrium conditions (a tuple), for an agent in
state `s` choosing `x`, given current (`y`, `z`) and future (`Y`, `Z`)
aggregates. To be defined for each model.
"""
function equilibrium end

function equilibrium(model::HModel, s::SVector, x::SVector, y::NamedTuple, z::NamedTuple, Y::NamedTuple, Z::NamedTuple)
    ss = NamedTuple{variables(model.agent.states)}(s)
    xx = NamedTuple{variables(model.agent.controls)}(x)
    SVector(equilibrium(model, ss, xx, y, z, Y, Z)...)
end


###
### Setting the agent's exogenous variables
###

# replace the entries of `q` (with names `vars`) that appear in `p`
@inline function _update(vars::NTuple{k,Symbol}, q::SVector{k,T}, p::NamedTuple) where k where T
    SVector{k,T}(ntuple(i -> convert(T, get(p, vars[i], q[i])), Val(k)))
end

function _update(calib::NamedTuple{names}, p::NamedTuple) where names
    NamedTuple{names}(map(k -> convert(typeof(calib[k]), get(p, k, calib[k])), names))
end

"""
    set_exogenous(agent, p::NamedTuple)

Returns a copy of the agent model where the exogenous variables listed in `p`
take the given values (in the exogenous process, the state space and the
calibration). The returned model has the same type as `agent`.
"""
function set_exogenous(agent::Dolo.YModel{N, <:Dolo.MarkovChain}, p::NamedTuple) where N
    mc = agent.exogenous
    vars = variables(mc)
    Q = map(q -> _update(vars, q, p), mc.Q)
    exogenous = typeof(mc)(mc.P, Q)
    exo_space = typeof(agent.states.exo)(Q)
    states = Dolo.ProductSpace((exo_space, agent.states.endo))
    calibration = _update(agent.calibration, p)
    return typeof(agent)(states, agent.controls, exogenous, calibration, agent.source)
end

set_exogenous(agent, p) = error("set_exogenous not implemented for agents with exogenous process $(typeof(agent.exogenous))")

"""
    agent_model(model::HModel, y=calibrated aggregates, z=calibrated exogenous)

The agent's model, given aggregate values `y` and `z` (via `projection`).
"""
function agent_model(model::HModel, y::NamedTuple=calibrated_aggregate(model), z::NamedTuple=calibrated_exogenous(model))
    p = projection(model, y, z)
    set_exogenous(model.agent, p)
end


function Base.show(io::IO, m::HModel)
    println(io, "Heterogeneous Agents Model: ", name(m))
    println(io, "* agent: ", name(m.agent), " (states: ", join(variables(m.agent.states), ", "),
        "; controls: ", join(variables(m.agent.controls), ", "), ")")
    println(io, "* aggregate: ", join(get_aggregate(m), ", "))
    print(io, "* exogenous: ", typeof(m.exogenous).name.name, "(", join(get_exogenous(m), ", "), ")")
end
