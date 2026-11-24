module Dolark

    using StaticArrays
    using LinearAlgebra
    using SparseArrays

    import Dolo
    import Dolo: variables, name, discretize, time_iteration

    include("model.jl")
    include("linsolve.jl")
    include("distribution.jl")
    include("steady_state.jl")
    include("derivatives.jl")
    include("agent_jacobian.jl")
    include("agent_pass.jl")
    include("joint.jl")
    include("egm.jl")

    export HModel, projection, equilibrium
    export agent_model, ergodic_distribution, transition_matrix, aggregate
    export excess, solve_agent, solve_steady_state, excess_jacobian, solve_steady_state_joint
    export egm, EGMResult, cold_start,
    poststates, half_transition, expectation, direct_response_egm, reverse_state, control_bounds,
    half_transition_depends_on_m,
    agent_jacobian, newton_sparse, agent_pass

end # module
