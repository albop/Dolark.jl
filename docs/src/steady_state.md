# Steady-state algorithms

Dolark has two solvers for the stationary equilibrium of a heterogeneous-agent
model:

| | function | unknowns of the outer iteration | the agent's problem is... |
|---|---|---|---|
| **nested** | `solve_steady_state` | aggregates `y` only | solved accurately at every trial value of `y` |
| **joint** | `solve_steady_state_joint` | controls `x`, distribution `μ` and aggregates `y` together | never solved on its own: it converges together with `y` |

Both are Newton methods with exact derivatives. They solve the same discretized
equations and give the same solution, up to the tolerances. Each joint
iteration costs about as much as one Jacobian evaluation of the nested solver,
but it does not re-solve the agent's problem. For this reason the joint solver
is usually the faster of the two. The nested solver is more conservative, since
every iterate is an exact agent solution at the current prices.

The fast code paths described below require:
- a one-dimensional endogenous grid, either uniform or any of Dolo's 1-d grids
  (`LogGrid`, `AssetGrid`, ...);
- a Markov chain for the agent's exogenous process;
- linear interpolation of the decision rule.

For other models, the solvers fall back on GMRES with Dolo's linear operators
(see [Fallbacks](@ref steady-state-fallbacks)).

## The discretized equations

The agent's grid has ``N = n_m n_s`` points ``s_n``: ``n_m`` exogenous nodes
times ``n_s`` endogenous nodes. The exogenous index varies fastest. Aggregates
``y`` (e.g. capital) and aggregate exogenous variables ``z`` determine prices
``p = \mathrm{projection}(y, z)`` (e.g. ``r, w``). The prices enter the agent's
problem as the values of some of its exogenous variables.

**Agent.** Controls ``x = (x_n)_{n}``, with ``x_n \in \mathbb{R}^{n_x}``. Next
period's endogenous state, at exogenous node ``j``, is
``S_{nj} = g(s_n, x_n, M_j(p))``, where ``M_j(p)`` are the values of the
exogenous variables at node ``j``. Let ``k_{nj}`` be the grid cell containing
``S_{nj}``, and ``\lambda_{nj}`` the barycentric weight of ``S_{nj}`` in that
cell. Future controls are interpolated linearly:

```math
X_{nj} = (1-\lambda_{nj})\, x_{(j, k_{nj})} + \lambda_{nj}\, x_{(j, k_{nj}+1)}
```

Outside the grid, the first or last cell is used and ``\lambda`` falls outside
``[0, 1]`` (linear extrapolation). The residuals are Dolo's, with the
complementarity conditions applied to the expectation of the arbitrage
equations:

```math
F_n(x; p) = \mathrm{comp}\Big(s_n, x_n, \sum_j P_{i(n) j}\, f(s_n, x_n, S_{nj}, X_{nj})\Big)
```

**Distribution.** Mass at ``s_n`` moves to node ``j`` with probability
``P_{i(n)j}``. It is then split between ``(j, k_{nj})`` and ``(j, k_{nj}+1)``
with weights ``1-\bar\lambda_{nj}`` and ``\bar\lambda_{nj}``, where
``\bar\lambda = \min(\max(\lambda, 0), 1)``. These are Young's lotteries: all
the mass beyond the grid goes to the closest bound. This defines the
transition matrix ``\Pi(x, p)``. The stationary distribution solves
``\mu = \Pi'\mu``, ``\sum_n \mu_n = 1``.

**Equilibrium.** ``A(\mu, x, y) = \sum_n \mu_n\, e(s_n, x_n, y, z) = 0``, where
``e`` is the model's `equilibrium` function (e.g. savings minus capital).

## Building blocks

### One pass over the grid (`agent_pass`)

Everything the solvers need about the agent at ``(x, p)`` comes from a single
pass over the grid. At each grid point:

1. **Locate the future states.** The cell of each future state ``S_{nj}`` is
   found by walking from the cell found at the previous grid point, on
   precomputed nodes. Future states are usually monotone in the endogenous
   state, so this takes a step or two. The walk works for any order of the
   points; it is only slower when the states jump.
2. **Evaluate with dual numbers.** The residual ``F_n`` and the weights
   ``\lambda_{nj}`` are evaluated once with dual numbers with respect to
   ``u = (x_n, p, \Delta X_{1}, \dots, \Delta X_{K})``. Here ``\Delta X_j`` is
   a perturbation of the interpolated future control at node ``j``, and ``K``
   is the number of future exogenous nodes. This one evaluation gives:
   - ``F_x``, the derivative w.r.t. ``x_n``, including the dependence of
     ``S_{nj}`` and ``\lambda_{nj}`` on ``x_n``;
   - ``F_p``, the derivative w.r.t. prices;
   - ``G_{nj} = \partial F_n / \partial X_{nj}``, which gives the derivative
     w.r.t. future controls;
   - ``\partial\lambda_{nj}/\partial(x_n, p)``, set to zero where the lottery
     weight is clamped.

Without derivatives (`derivatives=false`), the pass only evaluates residuals
and lotteries. The model's own `transition`, `arbitrage` and
`complementarities` are used throughout, so nothing here is specific to a
given model.

From the pass, the following are assembled directly in compressed sparse
column format:
- **the agent's Jacobian** ``M = F_x + F_X`` (`jacobian_matrix`). Row ``n``
  holds the ``n_x \times n_x`` block ``F_x`` on the diagonal, and
  ``(1-\lambda_{nj}) G_{nj}``, ``\lambda_{nj} G_{nj}`` in the columns of
  ``(j, k_{nj})`` and ``(j, k_{nj}+1)``. So there are at most ``1 + 2K``
  blocks per row;
- **the transition matrix** ``\Pi``, and the derivatives ``\Pi_x``, ``\Pi_p`` of
  ``v \mapsto \Pi(x,p)'\mu`` (`distribution_derivatives`);
- **the product** ``\Pi'\mu`` (`forward`), computed without forming ``\Pi``.

### Stationarity system (`DistributionSolver`)

The ``N`` equations ``(I - \Pi')\mu = 0`` have rank ``N-1``: they sum to zero,
so any one of them can be dropped. The equation of a point ``n_0`` that
carries mass is replaced by the normalization ``\mu_{n_0} = 1``. ``\mu`` is
then rescaled to sum to one.

This keeps the matrix ``A`` as sparse as ``\Pi``, with a unit diagonal. A dense
row ``\sum \mu = 1`` would make sparse factorizations several times slower, and
a zero on the diagonal breaks incomplete factorizations.

The normalization point ``n_0`` is the mode of the distribution at the
previous iterate. At the first iterate, it is the mode after 50 iterations of
``\mu \leftarrow \Pi'\mu`` from a uniform distribution. If the solve fails
(little mass at ``n_0``), the other choice is tried, and finally the dense
mass condition.

Linear systems ``(I-\Pi')d = r`` (all equations but the last) with
``\sum d = m`` are solved as follows. The last equation is implied by the
others, so equation ``n_0`` is dropped instead. The system is solved with
``d_{n_0} = 0``, and the null vector is then added:
``d \leftarrow d + (m - \textstyle\sum d)\,\mu``.

### Sparse linear systems (`SparseSolver`)

``M`` and ``A`` are ``N n_x \times N n_x`` and ``N \times N`` matrices, with
``O(1 + 2K)`` entries per row. They are solved by GMRES (Krylov dimension 60),
right-preconditioned by an incomplete LU factorization (`IncompleteLU.jl`,
drop tolerance ``\tau = 10^{-2}``). The factorization builds in about 1 ms for
``N = 3500``, and GMRES then converges in 10 to 25 iterations. That is several
times cheaper than an exact sparse LU, whose fill-in is large here: the
transitions are not banded.

If GMRES fails, the matrix is factorized exactly (UMFPACK), and that
factorization is kept for later solves. The tolerance is relative to the
right-hand side. Newton steps use a loose tolerance (inexact Newton), and the
quantities that are themselves results use a tight one:

| system | relative tolerance |
|---|---|
| agent's Newton steps (nested) | ``10^{-8}`` |
| joint Newton steps (agent and distribution blocks) | ``10^{-6}`` |
| ergodic distribution and IFT Jacobian (nested) | ``10^{-12}`` |

### Initial guess: endogenous grid method (`egm`)

An agent can define `Dolark.cold_start(agent, dmodel)`, which returns an
initial guess for its controls. For the consumption-savings agent of the
Aiyagari example, it is `egm(dmodel)`: the endogenous grid method on a
log-spaced grid of savings, by default 100 points and tolerance ``10^{-3}``
on the change in consumption. The result is interpolated on Dolo's grid.

It does not solve Dolo's discretized problem exactly (the two
discretizations differ), so it is only a starting point for Newton's method.

## Nested solver: `solve_steady_state`

This is a damped Newton method on the ``n_y`` equations ``A(y) = 0``. Here
``A(y)`` is the excess function: at aggregates ``y``, the agent is solved
exactly, then the stationary distribution is computed, then the equilibrium
conditions are integrated (`excess`).

**1. Excess function `excess(model, y)`.**

1. Compute prices ``p(y)`` and discretize the agent's problem.
2. Solve ``F(x; p) = 0`` to ``\max|F| < 10^{-10}`` (`solve_agent`), with
   `newton_sparse`, a chord–Newton method:
   - a Newton step ``dx = -M^{-1}F`` is solved with a `SparseSolver` of the
     agent's Jacobian ``M``;
   - the matrix and its preconditioner are **reused** in the next iterations
     (chord method) as long as the residual falls by a factor of at least
     ``\theta = 0.25`` per step. Otherwise they are recomputed at the current
     point;
   - with a fresh Jacobian, steps are halved until the residual decreases
     (up to 10 times). A step with a stale Jacobian that does not decrease the
     residual triggers a new Jacobian instead.

   Newton starts from the controls predicted from the previous outer iterate
   (see step 3 below), with the previous iterate's Jacobian solver as its
   initial chord matrix. At the first evaluation, it starts from the cold
   start (EGM).
3. Compute the stationary distribution with a `DistributionSolver`, normalized
   at the mode of the previous iterate's distribution.
4. Compute ``A = \sum_n \mu_n e(s_n, x_n, y, z)``.

**2. Exact Jacobian `excess_derivatives`**, by the implicit function theorem
at the solution ``(x, \mu)``:

```math
\begin{aligned}
\frac{dx}{dy} &= -M^{-1} F_p\, p_y \\
(I - \Pi')\frac{d\mu}{dy} &= \Pi_x \frac{dx}{dy} + \Pi_p\, p_y, \qquad \textstyle\sum \frac{d\mu}{dy} = 0 \\
\frac{dA}{dy} &= A_\mu \frac{d\mu}{dy} + A_x \frac{dx}{dy} + A_p\, p_y + A_y
\end{aligned}
```

``F_p``, ``M``, ``\Pi_x`` and ``\Pi_p`` come from one `agent_pass` with
derivatives. ``A_\mu``, ``A_x``, ``A_p`` and ``A_y`` come from automatic
differentiation of the model's `equilibrium` function at each grid point. The
distribution system reuses the `DistributionSolver` of step 1.3. This takes
``n_y`` solves with ``M`` and ``n_y`` with ``A``.

**3. Newton step** ``dy = -(dA/dy)^{-1} A``, then backtracking.

1. Steps are halved (up to 10 times) until ``\max|A|`` decreases.
2. Iterates are clamped to the bounds of `model.aggregate`.
3. Each trial point ``y_1`` is warm-started with the first-order prediction
   ``x_1 = x + \frac{dx}{dy}(y_1 - y)``. Close to the solution this prediction
   is already accurate to the agent's tolerance, so no Newton step is needed.

**Stopping rule:** ``\max|A| < 10^{-6}`` (`tol`). At most 50 iterations.

`jacobian=:fd` replaces step 2 by forward finite differences of `excess`.

## Joint solver: `solve_steady_state_joint`

This is a damped Newton method on the full system in ``u = (x, \mu, y)``,
which has ``N n_x + N + n_y`` unknowns:

```math
R(u) = \begin{pmatrix}
R_x = F(x;\, p(y)) \\
R_\mu = \Pi(x, p(y))'\mu - \mu, \ \text{last entry replaced by } \textstyle\sum\mu - 1 \\
R_A = \sum_n \mu_n\, e(s_n, x_n, y, z)
\end{pmatrix}
```

**1. Initial point.**
- ``x_0`` is the agent's cold start (EGM) at ``y_0``, used as is. If no cold
  start is defined, the agent is solved first.
- ``\mu_0`` is the stationary distribution of ``x_0``.

**2. Newton step** (`joint_newton_step`, `linsolve=:direct`). The step solves
``J\,du = -R`` exactly, up to the GMRES tolerance, by block elimination. All
derivatives come from one `agent_pass`:

```math
\begin{aligned}
dx &= X_0 + X_1\, dy, & X_0 &= -M^{-1} R_x, & X_1 &= -M^{-1} F_p\, p_y \\
d\mu &= U_0 + U_1\, dy, & U_0 &= \mathcal{S}(R_\mu + \Pi_x X_0;\ -R_{\mu,N}), & U_1 &= \mathcal{S}(\Pi_x X_1 + \Pi_p\, p_y;\ 0) \\
dy &= -\big(A_x X_1 + A_\mu U_1 + A_y + A_p\, p_y\big)^{-1}\big(R_A + A_x X_0 + A_\mu U_0\big)
\end{aligned}
```

Here ``\mathcal{S}(r; m)`` solves ``(I-\Pi')d = r`` (all equations but the
last) with ``\sum d = m``, as described in the stationarity system section
above.

A step costs:
- one Jacobian ``M`` with its incomplete factorization, and ``1 + n_y`` GMRES
  solves with it;
- one stationarity system with its factorization, and ``1 + (1 + n_y)`` GMRES
  solves with it: the ergodic distribution, warm-started from the current
  ``\mu``, then ``U_0`` and ``U_1``;
- an ``n_y \times n_y`` dense solve.

The solve tolerance is ``10^{-6}``.

**3. Damping.**
1. The step is scaled so that no aggregate changes by more than 20% of
   ``\max(|y|, 1)`` (`max_step_y`).
2. Steps are then halved (up to 10 times) until ``\|R\|_\infty`` decreases.
   Aggregates are clamped to the bounds of `model.aggregate`, and points where
   the model is not defined (e.g. negative consumption, a `DomainError`) are
   rejected.

The residual at a trial point is computed from an `agent_pass` with
derivatives. If the point is accepted, which is the usual case, the next
Newton step reuses that pass.

**Stopping rule:** ``\|R\|_\infty < 10^{-8}`` (`tol`). This bounds the Euler
errors, the stationarity residuals and the equilibrium conditions together.
At most 30 iterations.

`linsolve=:gmres` (`agent_steps`, `krylov_dim`, `η`) replaces step 2 by GMRES
on the full linearized system (`joint_linearization`), applied matrix-free.
It is right-preconditioned by a block lower triangular approximation of
``J``. This is the default when the fast path is not available.

## [Fallbacks](@id steady-state-fallbacks)

- **Grids or decision rules not covered by the fast path** (more than one
  endogenous dimension, cubic interpolation): the agent is solved with Dolo's
  `newton`, using GMRES on Dolo's linear operators. The IFT Jacobian uses the
  preconditioned operator ``(I - L)``, with ``L = -F_x^{-1}F_X``. The joint
  solver uses `linsolve=:gmres`.
- **`solve_agent`:** if Newton fails from the warm start and from the cold
  start, it tries accelerated time iteration followed by Newton, then plain
  time iteration followed by Newton.
- **`SparseSolver`:** exact LU if GMRES fails.
- **`DistributionSolver`:** another normalization point, then the dense mass
  condition.

## Performance

These timings use the Aiyagari model with the baseline HABenchmark
calibration: ``\beta = 0.96``, EIS ``0.5``, ``\alpha = 0.36``,
``\delta = 0.08``, a 7-state Rouwenhorst income process, and 500 log-spaced
points, i.e. ``N = 3500``. The initial guess is ``K = 6``. Times are medians
of warm runs on an Intel i7-9700 (`examples/ayiagari_habenchmark.jl`):

| | time | iterations |
|---|---|---|
| nested (`solve_steady_state`) | ≈ 230 ms | 5 |
| joint (`solve_steady_state_joint`) | ≈ 90 ms | 6 |
| sequence-space Jacobian toolkit, steady state (Brent on ``r``) | 547 ms | 9 evaluations |

For the joint solver, a Newton step takes about 14 ms:
- the agent pass with derivatives, about 3 ms;
- the two sparse systems (incomplete factorizations and GMRES solves), about
  8 ms;
- assembly and the trial residual, the rest.

EGM takes about 5 ms. The number of outer iterations is driven by the
nonlinearity of ``A(K)``. That is why both solvers need 5–6 iterations from
``K = 6``, even though each iteration uses exact derivatives.
