# Dolark.jl

Heterogeneous agents models on top of [Dolo.jl](https://github.com/EconForge/Dolo.jl).

Models are written in pure Julia, as bits objects, following Dolo's conventions:

- the individual problem is a Dolo model (`Dolo.YModel`), whose exogenous
  variables include the prices set by the aggregate model;
- `HModel(name, agent, aggregate, exogenous, calibration)` adds the aggregate
  endogenous variables (a `CartesianSpace`), the aggregate exogenous process and
  the calibration;
- equations are methods dispatched on `HModel{name}`:
  - `projection(model, y, z)`: agent's prices given aggregates;
  - `equilibrium(model, s, x, y, z, Y, Z)`: individual contribution to the
    equilibrium conditions (integrated against the distribution of agents).

See `examples/ayiagari.jl` (the model) and `examples/ayiagari_steady_state.jl`:

```julia
using Dolark
model = include("examples/ayiagari.jl")
sol = solve_steady_state(model; discretization=Dict(:endo => Dict(:n => 200)))
sol.y      # (K = 14.829...,)
```

The YAML representation of `models/ayiagari.yaml` is not supported (yet).
