# Maze RL: Tabular Q-Learning to DQN to PPO

A native C and raylib teaching project that walks the same maze problem up three rungs of reinforcement learning, each written from scratch with no machine-learning library:

| Learner | Family | What it learns | Key machinery |
| --- | --- | --- | --- |
| Tabular Q-learning | value-based | a 100 by 4 Q-table | one update rule |
| DQN | value-based, off-policy | `Q(s,a)` via a network | replay buffer, target network |
| **PPO** | policy-gradient, on-policy | `pi(a\|s)` directly | GAE, clipped surrogate objective |

**The project's focus is PPO.** The first two rungs exist to make its advantage measurable: on a suite of mazes the agent never trains on, PPO roughly doubles DQN's held-out success at identical architecture, observation, environment and step budget.

<!-- For inline playback on github.com, drag assets/ppo_generalization.mp4 into any
     issue or PR comment, then paste the resulting https://github.com/user-attachments/assets/...
     URL on its own line in place of the link below. Repo-relative paths do not autoplay. -->

**[▶ Watch the demo (51s)]** — PPO training live, then greedy rollouts on training mazes, on unseen mazes, and on one it fails. 

https://github.com/user-attachments/assets/6569d253-5d22-47de-979f-c08835487a53


## Build and run

```bash
make            # macOS/Linux (expects raylib in /opt/homebrew, override RAYLIB_PREFIX)
./maze_rl
```

On Windows with MSYS2 raylib at `C:/msys64/mingw64`, `make` produces `maze_rl.exe`. The binary serves both the interactive raylib app and the headless experiments; no ML library is required.

## PPO essentials

Let `s_t` be the state at step `t`, `a_t` the action taken, `theta` the actor parameters and `phi` the critic parameters.

### The core difference from DQN

DQN learns values and *derives* behaviour from them (`argmax Q`, with epsilon bolted on for exploration). PPO learns the behaviour directly as a probability distribution, and exploration is just *sampling* it. The critic is not used to pick actions — only to judge how much better than expected an action turned out.

### Policy and rollouts

The actor emits four logits; a softmax turns them into probabilities:

```text
pi(a|s) = exp(z_a) / sum_b exp(z_b)
```

Training runs on-policy: collect a fixed 2,048-step rollout with the *current* policy, storing `(s, a, log pi(a|s), r, V(s), done)` at each step, update on it, then throw it away. There is no replay buffer — reusing old data would mean optimizing a policy using actions a different policy chose.

### Generalized Advantage Estimation

The advantage says how much better an action was than the critic expected. The one-step TD residual is:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

GAE blends residuals over the future with a decay `lambda`, computed backwards in one pass:

```text
A_t = delta_t + gamma * lambda * A_{t+1}        (A = 0 at episode end)
```

`lambda` trades bias against variance: `lambda = 0` is pure TD (trusts the critic, low variance, biased when the critic is wrong), `lambda = 1` is the full discounted return (unbiased, high variance). The ablations below show this parameter is load-bearing — and that the right value depends on the task, not on theory.

### The clipped surrogate objective

Reusing a rollout for several epochs means the policy drifts from the one that collected the data. The importance ratio measures the drift:

```text
r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
```

A plain `r_t * A_t` objective lets a single batch push the policy arbitrarily far. PPO takes the pessimistic branch:

```text
L_clip = mean_t[ min( r_t * A_t , clip(r_t, 1-eps, 1+eps) * A_t ) ]
```

When the ratio leaves `[1-eps, 1+eps]` in the *improving* direction the gradient becomes exactly zero, so the update stops paying for drift it can no longer trust. It is an intentionally one-sided brake: moves that make a good action *less* likely are never clipped away.

### Entropy bonus and the full objective

Entropy `H(pi) = -sum_a pi(a|s) log pi(a|s)` measures indecision. Adding it to the objective penalizes premature certainty, which matters here because a policy that collapses early stops finding the goal at all.

```text
L(theta, phi) = L_clip - c_v * mean_t[(V_phi(s_t) - R_t)^2] + c_ent * mean_t[H(pi_theta(.|s_t))]
```

Advantages are normalized to zero mean and unit variance per batch, which keeps the gradient scale stable.

### Hyperparameters and shapes

| Setting | Value | Flag |
| --- | ---: | --- |
| Rollout length | 2,048 steps | |
| Epochs per rollout (`K`) | 4 | `--epochs` |
| Minibatch | 256 | |
| Clip `eps` | 0.2 | `--clip-eps` |
| `gamma` / `lambda` | 0.95 / 0.95 | `--gae-lambda` |
| Value coef / entropy coef | 0.5 / 0.01 | `--entropy-coef` |
| Adam learning rate | 0.0003 | `--lr` |

Actor and critic are separate networks sharing no trunk, so each backward pass is independently verifiable by finite differences. On the single maze the actor matches the DQN exactly (one-hot input, one 64-unit hidden layer, 6,724 parameters); on the generalization suite it is 22,084 parameters, again matching its DQN counterpart.

## Results

### Single maze: opposite strengths

```bash
make ppo    # or: ./maze_rl --ppo --steps 60000 --seeds 5 --seed 1 --compare-dqn
```

| Algorithm | Optimal route | First greedy solve (env steps) | Wall clock / 60k steps |
| --- | --- | ---: | ---: |
| PPO | 5/5 seeds, 14 steps | 13,107 mean | ~120 ms |
| DQN | 5/5 seeds, 14 steps | 4,555 mean | ~990 ms |

Both find the same optimal route every seed. **DQN is ~2.9x more sample-efficient** — replay reuses each transition many times, while PPO discards each batch after 4 epochs. **PPO is ~8x faster in wall clock**, because DQN runs a small update every 4 environment steps while PPO runs far fewer, larger ones. Comparison is on environment steps, not episodes, since an episode means different amounts of experience to each algorithm.

### Held-out mazes: where PPO wins

The real test is 34 mazes — 16 trained on, 18 never seen. Both algorithms share one implementation of the maze suite, the layout encoding and the step function (`src/mazesuite.c`), so they provably see identical mazes and dynamics.

```bash
make ppo-generalization
```

Over **20 seeds** (360 held-out evaluations each):

| Algorithm | Parameters | Train-fit | Held-out |
| --- | ---: | ---: | ---: |
| DQN layout MLP | 22,084 | 49.1% | 8.9% |
| DQN conv (10 seeds) | 13,624 | 25.6% | 5.0% |
| DQN wide conv (10 seeds) | 21,809 | 45.0% | 5.6% |
| **PPO layout** | **22,084** | **49.4%** | **18.3%** |

**PPO roughly doubles DQN's held-out success at essentially identical train-fit** (49.4% vs 49.1%). That equality is the point: this is not a difference in how well each fits its training data, it is a difference in what transfers — produced by changing the algorithm alone.

Per-seed head-to-head is PPO 12 wins, DQN 4, 4 ties; a paired test gives **t = 3.41 (df = 19), p ≈ 0.003**; the gap survives dropping each side's best seed (17.0% vs 7.6%). The effect *strengthened* from 10 to 20 seeds (t rose 1.93 → 3.41), which is what a real effect does and noise generally does not.

### Ablations: what each component actually does

**Data reuse (`--epochs K`) is where sample efficiency comes from.** Single maze, 10 seeds:

| K | Steps to first solve | clip_fraction |
| ---: | ---: | ---: |
| 1 (no reuse) | 44,442 | 0.000 |
| 4 (default) | 13,312 | 0.012 |
| 10 | 7,578 | 0.039 |
| 20 | 6,963 | 0.061 |

K=1 is essentially vanilla policy gradient and needs **6.4x the steps** of K=20. `clip_fraction` is *exactly* zero at K=1, as it must be — on the only pass the ratio is identically 1, so clipping cannot bind. A free correctness check.

**Clipping does nothing until updates are aggressive — then it prevents total collapse.** Sweeping learning rate at K=10, measuring whether the final policy still solves:

| Learning rate | Clipping on | Clipping off | Final entropy (off) |
| ---: | ---: | ---: | ---: |
| 0.0003 (default) | 10/10 | 10/10 | 0.008 |
| 0.003 | 9/10 | **2/10** | 0.024 |
| 0.01 | 7/10 | **0/10** | **0.000** |

At lr=0.01 unclipped, entropy hits exactly 0.000 on all ten seeds. Seven of them *did* find the goal early and then destroyed their own policy: the unclipped objective kept raising the winning action's probability until nothing else could be sampled, and a deterministic policy on a wrong action cannot explore back out.

**GAE `lambda` must be high here — and the single-maze answer is the opposite of the real one.** On one fixed maze, low `lambda` wins monotonically. On the generalization suite it reverses:

| lambda | Train-fit | Held-out |
| ---: | ---: | ---: |
| 0 | 0.6% | **0.0%** |
| 0.5 | 21.9% | 10.6% |
| 0.95 (default) | 46.2% | 18.3% |
| 1.0 | 37.5% | 22.8% |

`lambda = 0` — the *best* single-maze setting — learns nothing on the real task. Low `lambda` means trusting the critic; across 16 mazes with randomized goals the critic is badly wrong early. (`lambda = 1.0` over the default is **not** established: t = 0.68.)

**The entropy bonus is load-bearing because exploration is the bottleneck.** Generalization suite, 10 seeds:

| c_ent | Train-fit | Held-out | Seeds learning nothing |
| ---: | ---: | ---: | ---: |
| 0 | 20.0% | 8.9% | **5 of 10** |
| 0.01 (default) | 46.2% | 18.3% | 1 of 10 |
| 0.05 | 61.9% | 20.0% | **0 of 10** |

Without the bonus half the seeds finish having learned nothing. With goals at separation ≥10 the goal is far away, so a policy that collapses early never receives the `+100` and has nothing to learn from. This is why *more* exploration pressure **raises** training fit instead of trading against it.

### Training past the peak makes generalization worse

Measured while checking a claim, not by design. Held-out eval every 50 updates, 3 seeds:

| seed | best held-out | at | at 1.84M steps |
| --- | ---: | ---: | ---: |
| 1 | 9/18 | 410k | 2/18 |
| 2 | 5/18 | 307k | 3/18 |
| 3 | 4/18 | 1.23M | 1/18 |

**Every seed peaks and then declines**, and none ends near its own best. At 1.9M steps the live dashboard reads **train return +33.5 against eval return −184** — still improving on the 16 training mazes while getting worse on the 18 unseen ones. Ordinary overfitting, in an RL policy, watchable live.

Two caveats. The peak *location* varies wildly: seed 3 sits at 0/18 solved with 18/18 livelocked until 716k steps, long after the others peaked. And the 575k budget used above was never chosen against a validation curve — it lands near-optimally for seeds 1 and 2 and truncates seed 3 before it learns anything, so some of the reported seed variance is plausibly an artifact of when training stops.

### A caveat that applies to every number above

Every result was scored against the **same 18 held-out mazes** from one hardcoded suite seed. No gradient ever touched them, but ~20 configurations have now been compared on them and the best kept — which makes the suite a validation set in practice.

From the observed per-seed spread, a 10-seed held-out mean carries **±4.1 pp** of standard error, and picking the best of 20 equally-good configurations inflates the winner by **~7.7 pp** on average.

- **Robust to this:** PPO vs DQN (9.4 pp, a single pre-specified comparison at 20 seeds, p ≈ 0.003, strengthening with more seeds), and the large qualitative results — `lambda = 0` failing outright, clipping preventing collapse, half the seeds learning nothing without entropy.
- **Not robust:** any "new best" worth a few points *selected* from a sweep. `lambda = 1.0`'s 22.8% is exactly the shape of number this manufactures for free, which is why it is not claimed over the default.

The suite seed has never been varied, so strictly these are statements about *these 34 mazes*. Re-running the headline comparisons on freshly generated suites is the highest-value outstanding experiment and has not been done.

## The 3D view

```bash
make train3d        # live training dashboard: return, KL, clip fraction, entropy, livelock
make render3d       # BFS shortest route, no learning
make ppo-demo-video # trains, renders frames, synthesizes the backing track, muxes
```

The dashboard plots approximate KL (Schulman's k3 estimator, `mean(exp(logr) - 1 - logr)` — non-negative and far lower variance than `mean(-logr)`), clip fraction, policy entropy and livelock rate alongside train and eval return. The backing track in the demo video is synthesized from scratch by `scripts/make_music.py`, so the video carries no third-party audio.

**Livelock** is the dominant failure mode worth knowing about: deterministic argmax can oscillate between two cells even when the policy can see the goal — at A the best action leads to B and at B back to A. Any exploration escapes it, so it is a property of greedy evaluation rather than of the policy being lost, and it is counted separately from "did not solve".

## How the earlier rungs got here

All learners share the reward structure (`+100` goal, `-5` wall bump, `-1` step), the 200-step limit and `gamma = 0.95`.

**Tabular Q-learning** is one update rule over 400 numbers:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

**DQN** replaces the table with a network and adds the two stabilizers that makes necessary — a frozen target network and a replay buffer — trained on Huber loss with Adam.

On the single fixed maze both reach the same optimal 14-step route on 10/10 seeds; the Q-table does it in 1.6 ms against DQN's 2,900 ms. That comparison is deliberately unfair in an instructive way: a table is an excellent fit for a tiny fully-observable environment, so the network buys nothing. Its value only appears on mazes the agent has never seen.

The DQN generalization experiments (`--procedural`, `--random-goals`, `--min-separation`, `--wide-conv`) established the ground PPO is measured on. The key finding: the original fixed-corner setup let agents learn "walk toward the bottom-right" rather than how to navigate, and removing that shortcut cut train-fit from 95% to 46% while held-out performance held steady. PPO reproduces this independently — its own train-fit falls 95.0% → 46.2% with random goals while held-out stays flat.

Interactive controls for the tabular/DQN app: `A` switch learner, `H` human, `T` train, `D` greedy demo, `C` clear, `R` reset, `P` policy arrows, `V` value heatmap, `+`/`-` speed. PPO is headless-only — its sampled stochastic policy does not fit the `learner.h` interface built around `selectAction(state, epsilon, rng)`.

## Tests

```bash
make test
```

PPO-side coverage, since hand-rolled policy gradients are easy to get subtly wrong:

- finite-difference gradient checks for actor and critic on both the one-hot and dense 340-input networks, probing an output-layer *and* a hidden-layer weight so an error in either backprop stage is caught
- GAE reduces to the one-step TD residual at `lambda = 0` and the discounted return at `lambda = 1`, and does not chain across episode boundaries
- softmax/log-prob consistency, including logits large enough to overflow a naive `exp`, and uniform entropy equal to `log(4)`
- the clipped branch contributes exactly zero policy gradient while the unclipped branch does not
- determinism: the same seed produces bit-identical networks after a full collect/GAE/train cycle

Shared and DQN-side: environment transitions, replay wraparound, terminal targets, target-network copying, procedural maze solvability, finite-difference checks on the convolutional models, and end-to-end learning of the optimal route.

## Source layout

```text
src/
|-- main.c               Interactive UI and command-line dispatch
|-- ppo.c/.h             PPO: actor/critic, rollouts, GAE, clipped objective
|-- mazesuite.c/.h       Shared maze suite, layout encoding and step function
|-- render3d.c/.h        3D view, live training dashboard, video director
|-- generalization.c/.h  Multi-maze held-out DQN experiments
|-- dqn.c/.h             Network, replay, target network, Adam
|-- tabular.c/.h         Tabular Q-learning
|-- learner.c/.h         Common learner interface (tabular and DQN)
|-- environment.c/.h     Shared rewards and transitions
|-- trainer.c/.h         Shared episode and metric logic
|-- benchmark.c/.h       Seeded headless comparison and CSV export
|-- rng.c/.h             Deterministic project RNG
|-- maze.c/.h            Maze state and rendering
`-- agent.c/.h           Movement and model-independent visualization
```

`ppo.c` is self-contained rather than implementing `learner.h`. It shares the *environment* with the DQN experiments through `mazesuite.h` — so both algorithms provably see identical mazes, observations and dynamics — while each keeps the training loop its own paradigm needs.
