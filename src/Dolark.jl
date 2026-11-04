module Dolark

    using StaticArrays
    using LinearAlgebra
    using SparseArrays

    import Dolo
    import Dolo: variables, name, discretize, time_iteration

    include("model.jl")
    include("distribution.jl")
    include("steady_state.jl")
    include("derivatives.jl")
    include("joint.jl")

    export HModel, projection, equilibrium
    export agent_model, ergodic_distribution, transition_matrix, aggregate
    export excess, solve_agent, solve_steady_state, excess_jacobian, solve_steady_state_joint

end # module
