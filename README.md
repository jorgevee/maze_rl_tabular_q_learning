# Maze RL: Tabular Q-Learning to DQN to PPO

A native C and raylib teaching project that walks the same maze problem up three rungs of reinforcement learning, each written from scratch with no machine-learning library:

| Learner | Family | What it learns | Key machinery |
| --- | --- | --- | --- |
| **Tabular Q-learning** | value-based | a 100 by 4 Q-table | one update rule |
| **DQN** | value-based, off-policy | `Q(s,a)` via a network | replay buffer, target network |
| **PPO** | policy-gradient, on-policy | `pi(a\|s)` directly | GAE, clipped surrogate objective |

![Trained Q-learning maze showing policy arrows and the Q-value heatmap](assets/mazesc1.png)

Each rung answers something the previous one couldn't. The tabular-vs-DQN comparison is deliberately unfair to DQN in an instructive way: a Q-table is an excellent fit for this tiny, fully observable environment, so DQN reaches the same optimal policy at far greater cost and no benefit. The point of the neural network only becomes visible in the [held-out generalization experiment](#held-out-maze-generalization-experiment), where the agent must handle mazes it never trained on. PPO then changes the *algorithm* rather than the data or architecture — and, on that same suite, roughly doubles DQN's held-out success.

**Current focus is PPO** — see [PPO essentials](#ppo-essentials) for the math, and [`ppo_design.md`](ppo_design.md) for the design and open questions.

## Build and run

The Makefile expects the MSYS2 raylib installation at `C:/msys64/mingw64`. If GNU Make is installed:

```powershell
make
.\maze_rl.exe
```

The executable supports both the interactive raylib application and headless experiments. No external machine-learning library is required.

On the current Windows setup, where `gcc` is available but `make` is not on `PATH`, use:

```powershell
gcc -std=c11 -O2 -Wall -Wextra -pedantic src\main.c src\maze.c src\agent.c src\environment.c src\rl.c src\rng.c src\learner.c src\tabular.c src\dqn.c src\ppo.c src\trainer.c src\benchmark.c src\generalization.c -o maze_rl.exe -IC:\msys64\mingw64\include -LC:\msys64\mingw64\lib -lraylib -lopengl32 -lgdi32 -lwinmm -lm
.\maze_rl.exe
```

## The three algorithms at a glance

All learners share the maze, the reward structure (`+100` goal, `-5` wall bump, `-1` step), the 200-step episode limit, and the discount factor `gamma = 0.95`.

**Tabular Q-learning** — one update rule, 400 numbers, no network:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

**DQN** — the same Bellman idea with a network standing in for the table, plus the two stabilizers that makes necessary:

```text
y_j = r_j                                       if terminal
y_j = r_j + gamma * max_a' Q(s'_j, a'; theta^-) otherwise    <- frozen target network

L(theta) = mean over minibatch of Huber_1( y_j - Q(s_j, a_j; theta) )
```

- 100-element one-hot state input, `100 -> 64 ReLU -> 4` (6,724 parameters), He initialization
- 10,000-transition circular replay buffer, 500-transition warm-up, 32-transition minibatches
- Adam at learning rate `0.001`, gradient-norm clipping at `10`
- target network copied every 250 optimizer updates
- epsilon-greedy exploration, `1.0` decaying to `0.05`

One-hot input keeps the baseline honest: DQN gets the same state identity the Q-table uses, without pretending flattened state numbers have meaningful numeric distance.

**PPO** — a different family entirely; the math is below.

The DQN training flow is available as a Graphviz diagram in [`assets/dqn_algorithm.dot`](assets/dqn_algorithm.dot):

```powershell
dot -Tsvg assets/dqn_algorithm.dot -o assets/dqn_algorithm.svg
```

## PPO essentials

Let `s_t` be the state at step `t`, `a_t` the action taken, `theta` the actor (policy) parameters, and `phi` the critic (value) parameters.

### The core difference from DQN

DQN learns `Q(s,a)` — a number per action — and *derives* behavior by taking the argmax, with an epsilon schedule bolted on from outside so it explores. PPO learns the behavior directly:

```text
DQN:  network -> Q(s,a) for each a       -> act = argmax  (+ epsilon noise)
PPO:  network -> pi(a|s), a distribution -> act = sample from it
```

Exploration is therefore *intrinsic* to PPO. An uncertain policy spreads probability across several actions and naturally tries them; a confident one doesn't. There is no epsilon anywhere in the implementation.

### Policy: softmax over action logits

The actor emits one logit per action, and a softmax turns them into a distribution:

```text
z = Actor(s; theta)                     <- 4 logits
pi(a|s)     = exp(z_a) / sum_b exp(z_b)
log pi(a|s) = z_a - log sum_b exp(z_b)
```

Implemented with the standard max-subtraction trick so large logits cannot overflow `exp`:

```text
m = max_b z_b
pi(a|s) = exp(z_a - m) / sum_b exp(z_b - m)
```

### Rollout collection

PPO is *on-policy*: it must learn from data its current policy generated, so there is no replay buffer. It collects a fixed batch of `T` environment steps (spanning several episodes), recording per step:

```text
a_t        ~ pi(.|s_t)              <- sampled, not argmax
logp_old_t = log pi(a_t|s_t)        <- frozen here; the "old policy" for the ratio below
v_t        = Critic(s_t; phi)
r_t, done_t                          <- from the environment
```

After a few epochs of updates the batch is **discarded**. That is PPO's fundamental sample-efficiency cost relative to DQN's replay, and it shows up directly in the single-maze result below (~2.9x more environment steps to first solve).

### Generalized Advantage Estimation (GAE)

The advantage answers "was this action better than what the critic expected from this state?" The cheapest estimate is the one-step TD residual:

```text
delta_t = r_t + gamma * v_{t+1} * (1 - done_t) - v_t
```

Using only `delta_t` is low-variance but biased — it trusts the critic completely. Summing all actual future rewards instead is unbiased but high-variance. GAE interpolates between those with `lambda`, as a backward recursion over the batch:

```text
A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
  A_T = 0     (nothing was collected beyond the last step)

lambda = 0  ->  A_t = delta_t                     (one-step TD: biased, low variance)
lambda = 1  ->  A_t = discounted return - v_t     (Monte Carlo: unbiased, high variance)
```

The critic's regression target falls out of the same quantity:

```text
target_t = A_t + v_t
```

Advantages are then normalized across the batch — standard and load-bearing, not cosmetic:

```text
A <- (A - mean(A)) / (std(A) + 1e-8)
```

### The clipped surrogate objective

This is the part PPO is named for. Because one batch is reused for several epochs, the policy drifts away from the one that collected the data. The importance ratio measures that drift:

```text
ratio_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
        = exp( log pi_new(a_t|s_t) - logp_old_t )
```

A plain policy-gradient objective `ratio_t * A_t` would happily push that ratio arbitrarily far within a single batch, which destabilizes training. PPO instead takes the *pessimistic* of the raw and clipped versions:

```text
L_clip_t = min( ratio_t * A_t,
                clip(ratio_t, 1 - eps, 1 + eps) * A_t )
  eps = 0.2
```

What that `min` actually does, case by case:

```text
A_t > 0 (good action), ratio > 1+eps  ->  clipped branch wins, gradient = 0
                                          (already made it likelier; stop)
A_t > 0,               ratio < 1-eps  ->  raw branch wins, gradient flows
                                          (drifted the wrong way; recover)
A_t < 0 (bad action),  ratio < 1-eps  ->  clipped branch wins, gradient = 0
                                          (already made it rarer; stop)
A_t < 0,               ratio > 1+eps  ->  raw branch wins, gradient flows
                                          (drifted the wrong way; push down)
```

So the clip only ever removes the incentive to keep moving in a direction already moved too far — it never blocks a correction back toward the old policy. In the implementation this is literally "if the clipped branch is selected, contribute no policy gradient," which a self-test verifies directly.

### Entropy bonus

Sampling alone doesn't guarantee continued exploration: a policy can collapse to near-deterministic early and stop discovering anything. An entropy term pushes back on that.

```text
H(s) = -sum_a pi(a|s) * log pi(a|s)

  H = log(4) ~ 1.386   uniform over 4 actions (maximum)
  H -> 0               deterministic
```

### The full objective

```text
policy_loss = -( mean_t[ L_clip_t ] + c_ent * mean_t[ H(s_t) ] )
value_loss  =    mean_t[ Huber_1( target_t - Critic(s_t; phi) ) ]

  c_ent = 0.01
```

Actor and critic are separate networks with separate Adam optimizers, so the two losses never mix gradients. A shared trunk is more common in production PPO; separate networks were chosen here so each backward pass is independently verifiable by finite differences.

### The update loop

```text
repeat:
  collect T = 2048 environment steps with the current policy
  compute GAE advantages and targets, then normalize the advantages
  for epoch in 1..K:                       K = 4
    shuffle the batch
    for each minibatch of 64:
      Adam step on the actor  using policy_loss
      Adam step on the critic using value_loss
  discard the batch
```

### Hyperparameters and network shapes

| Symbol | Meaning | Value |
| --- | --- | ---: |
| `gamma` | discount | 0.95 |
| `lambda` | GAE smoothing | 0.95 |
| `eps` | clip range | 0.2 |
| `c_ent` | entropy coefficient | 0.01 |
| `T` | rollout length (steps per batch) | 2,048 |
| `K` | epochs per batch | 4 |
| | minibatch size | 64 |
| `alpha` | Adam learning rate | 3e-4 |
| | gradient-norm clip | 0.5 |

Network shapes are chosen to match their DQN counterparts exactly, so no comparison is secretly about network size:

```text
single maze       actor:  one-hot(100) -> 64 ReLU -> 4 logits     6,724 params
                  critic: one-hot(100) -> 64 ReLU -> 1 value      6,529 params

generalization    actor:  crop(340)    -> 64 ReLU -> 4 logits    22,084 params
                  critic: crop(340)    -> 64 ReLU -> 1 value     21,889 params
```

The single-maze actor's 6,724 parameters match the single-maze DQN; the generalization actor's 22,084 match the layout-aware DQN.

### What each piece is actually for

A short map from machinery to the problem it solves, since PPO has more moving parts than DQN and it's easy to lose track of why. Each of these claims is tested empirically in [PPO ablations](#ppo-ablations-what-each-component-actually-does) further below — two of them turned out to be inert at the default settings.

| Piece | Problem it solves |
| --- | --- |
| Stochastic policy | exploration, without an external epsilon schedule |
| Critic + advantages | reduces gradient variance versus raw returns |
| GAE `lambda` | dials the bias/variance tradeoff in the advantage estimate |
| Importance ratio | lets one batch be reused for several gradient epochs |
| Clipping | stops that reuse from moving the policy too far off-batch |
| Entropy bonus | stops premature collapse to a deterministic policy |
| Advantage normalization | keeps gradient scale stable across batches |

## Interactive controls

| Input | Action |
| --- | --- |
| Arrow keys | Move the agent in human mode |
| `A` | Switch between the tabular and DQN learners |
| `H` | Enter human-control mode |
| `T` | Start or pause training for the selected learner |
| `D` | Demonstrate the selected learner's frozen greedy policy |
| `C` | Clear the selected model and its training state |
| `R` | Reset the visible agent position |
| `P` | Toggle learned-policy arrows |
| `V` | Toggle the Q-value heatmap |
| `+` / `-` | Adjust episodes trained per frame |

Switching learners preserves both models, their episode counts, and their metrics. Demonstrations use `epsilon = 0` and never update the model or replay buffer.

PPO is headless-only and does not appear in the interactive app. Its `learner.h`-incompatible shape (a sampled stochastic policy with no epsilon, trained on batched rollouts rather than one transition at a time) made an adapter more trouble than it was worth — see [`ppo_design.md`](ppo_design.md).

## Reproducible comparison

Run the default experiment (both learners, 5,000 episodes, 10 seeds):

```powershell
make benchmark
```

Or configure it directly:

```powershell
.\maze_rl.exe --benchmark --agent both --episodes 5000 --seeds 10 --seed 1 --csv comparison.csv
```

`--agent` accepts `tabular`, `dqn`, or `both`. The CSV records checkpoints every 100 episodes, including rolling training success, average successful path length, frozen greedy evaluation, elapsed CPU time, parameter count, and learner memory. Evaluation has its own deterministic RNG and therefore cannot change the subsequent training trajectory.

### Verified 10-seed result

The default 5,000-episode comparison was run for seeds 1 through 10:

| Agent | Optimal greedy runs | Greedy steps | Parameters | Learner memory | Mean CPU time/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tabular | 10 / 10 | 14 | 400 | 1,608 bytes | 1.6 ms |
| DQN | 10 / 10 | 14 | 6,724 | 307,628 bytes | 2,900.3 ms |

Both agents had solved the maze by the first 100-episode evaluation checkpoint in every seed. The result illustrates the intended lesson: DQN reaches the same policy, but this fixed discrete maze gives it no generalization benefit to offset its substantially greater computation and memory.

## Held-out maze generalization experiment

The single-maze DQN cannot genuinely generalize because its one-hot input contains only the agent's position. The same position in two different mazes produces the same observation even when walls require different actions. A separate headless experiment tests that limitation, a more informative observation, and an architecture with spatial weight sharing:

- **Position-only baseline:** a 144-element one-hot position in a padded 12 by 12 space.
- **Layout-aware MLP:** a 13 by 13 agent-centered crop with wall and goal channels, plus normalized goal displacement. This produces 340 inputs and works with every tested size up to 12 by 12.
- **Convolutional DQN:** the same layout observation processed by two shared 3 by 3 convolution stages and a 64-unit Q-value head.

The deterministic suite contains:

- 16 training mazes, all 10 by 10
- 6 unseen 10 by 10 mazes to test new layouts
- 6 unseen 8 by 8 mazes to test smaller layouts
- 6 unseen 12 by 12 mazes to test larger layouts

Every generated maze is validated with breadth-first search and its optimal route length is recorded. Training samples only the 16 training mazes; held-out mazes never enter replay and never update the network.

Run the default three-seed experiment with:

```powershell
.\maze_rl.exe --generalization --episodes 5000 --seeds 3 --seed 1 --csv generalization.csv
```

or, with GNU Make:

```powershell
make generalization
```

The CSV includes the reproducible maze-generation seed, split, dimensions, BFS-optimal steps, success, actual steps, optimality gap, return, training time, and model size.

### Three-seed result and interpretation

A 5,000-episode run with learner seeds 1 through 3 produced:

| Observation | Training 10 by 10 | Unseen 10 by 10 | Unseen 8 by 8 | Unseen 12 by 12 | All unseen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Position only | 0 / 48 | 0 / 18 | 0 / 18 | 0 / 18 | 0 / 54 |
| Layout-aware MLP | 48 / 48 | 1 / 18 | 1 / 18 | 0 / 18 | 2 / 54 |
| Convolutional | 48 / 48 | 5 / 18 | 6 / 18 | 1 / 18 | 12 / 54 |

The MLP result is evidence of memorization. Convolution raised held-out success from 3.7% to 22.2% while reducing parameter count from 22,084 to 13,624. It still failed most held-out evaluations, especially larger mazes.

**Update:** the "spatial weight sharing is a better inductive bias" reading of this gap does not survive a follow-up control. Both mazes in this baseline always place the goal at the same corner relative to the start, and the `--random-goals` experiment further below shows the convolutional model's entire held-out advantage disappears once that shared corner-to-corner direction is removed, while wall-layout diversity and architecture are held fixed. The gap here looks substantially explained by the convolutional model exploiting that shared direction more effectively than the MLP, not by superior maze-solving ability.

### Procedural training distribution

The 12/54 result above is capped by the training distribution as much as by the architecture: all 16 training mazes are 10 by 10 with the start pinned to `(1,1)` and the goal to the opposite corner, so the network can partly learn "head down and right" instead of "find a path to wherever the goal is." `--procedural` replaces that fixed set with a freshly generated maze every training episode, with a random size in `[--min-size, --max-size]` and a random start and goal cell (instead of fixed corners) each time. The held-out evaluation suite is untouched, so results are directly comparable to the fixed-training baseline above.

```powershell
.\maze_rl.exe --generalization --episodes 5000 --seeds 3 --seed 1 --procedural --min-size 6 --max-size 12 --regen-every 1 --csv generalization_procedural.csv
```

or with GNU Make:

```powershell
make generalization-procedural
```

`--regen-every N` (default 1) generates a new procedural maze every `N` episodes instead of every episode. The CSV gains a `train_mode` column (`fixed_16` or `procedural_<min>to<max>`) so fixed and procedural runs can sit in the same file. Passing no `--procedural` flag reproduces the exact fixed-training run above bit-for-bit (verified: identical success/steps/return/parameters per maze, only wall-clock training time differs).

**Result: at the same 5,000-episode budget, procedural training made held-out performance worse, not better.**

| Training regime | Held-out 10x10 | Held-out 8x8 | Held-out 12x12 | All held-out |
| --- | ---: | ---: | ---: | ---: |
| Fixed 16 mazes (baseline) | 5/18 | 6/18 | 1/18 | 12/54 |
| Procedural, fresh maze every episode | 0/18 | 1/18 | 0/18 | 1/54 |
| Procedural, fresh maze every 300 episodes (~17-maze pool) | 0/18 | 0/18 | 0/18 | 0/54 |

Both convolutional runs used identical episode budget, architecture, and held-out suite. Adding a diagnostic ("pool fit": greedy success on the mazes actually seen during training, evaluated by the final network) explains why:

- **Fresh-every-episode:** ~42% pool fit (108-115/256 sampled early-training mazes) despite ~2% on the fixed suite. The network does learn something transferable about navigating this procedural distribution, it just doesn't transfer to the fixed suite's specific task.
- **300-episode pool (~17 mazes, ~300 exposures each):** only 2-4/17 (12-24%) pool fit. This is the sharper finding — the fixed baseline hits 48/48 (100%) on its 16 training mazes at a similar exposure count, but the procedural pool's network mostly fails to fit even its *own* 17-maze training set.

The difference is what varies between mazes, not how many there are. The 16 fixed training mazes all share one start-to-goal vector (always `(1,1)` to the opposite corner) and differ only in wall layout, so the network can partly rely on a shared "go down-and-right" direction across all 16. Procedural generation randomizes size *and* start/goal per maze, so each of the 17 pool mazes is a distinct start-to-goal task with no shared direction to exploit — a much harder multi-task problem at the same gradient budget. On top of that, uniformly random start/goal cells are typically much closer together than the fixed suite's opposite-corner placement (mean Manhattan distance on a 10x10 grid is ~5.3 for a random pair versus 14 for opposite corners), so the procedural distribution under-trains exactly the long-path routing the fixed suite tests.

Net finding: naively broadening the training distribution (random size + random start/goal, unchanged episode budget) is not a free win here — it trades a learnable, narrow task for a much harder multi-task one without enough additional training signal to compensate.

### Isolating "random starts alone": `--random-goals`

The procedural result above confounds two changes at once: varying maze size and wall layout, and varying start/goal. `--random-goals` isolates the second: it keeps the same 16 fixed 10x10 wall layouts (so wall-layout diversity is exactly what the original baseline used) and only randomizes the start and goal cell on them, every episode.

```powershell
.\maze_rl.exe --generalization --episodes 5000 --seeds 3 --seed 1 --random-goals --csv generalization_random_goals.csv
```

`--random-goals` and `--procedural` are mutually exclusive (the CLI rejects passing both). The held-out suite evaluates each maze at its own fixed, canonical corner-to-corner start/goal regardless of training mode, so a `train` split score below 48/48 here means the network no longer solves the corner-to-corner task on mazes it trained on, only under a different start/goal each time.

| Training regime | Train-set fit (canonical corner task) | All held-out |
| --- | ---: | ---: |
| Fixed 16, fixed corners (baseline) — conv | 48/48 | 12/54 (22.2%) |
| Fixed 16, fixed corners (baseline) — layout MLP | 48/48 | 2/54 (3.7%) |
| Procedural, fresh maze every episode — conv | 2/48 | 1/54 (1.9%) |
| Fixed 16 walls, random start/goal — conv | 1/48 | 2/54 (3.7%) |
| Fixed 16 walls, random start/goal — layout MLP | 9/48 | 2/54 (3.7%) |

This isolates the effect cleanly and revises a conclusion from the convolutional-follow-up result above. With wall-layout diversity held constant at 16 mazes and only start/goal randomized, the convolutional model's held-out advantage over the layout-aware MLP **disappears** (2/54 versus 2/54, both at 3.7%), even though its raw capacity and architecture are unchanged. Since the fixed-corner baseline's convolutional advantage (22.2% vs 3.7%) survived nothing else changing except the maze's start/goal correlation, that advantage looks like it was substantially explained by the convolutional model exploiting the shared "always corner-to-corner" direction across the 16 training mazes more effectively than the MLP did — not by superior maze-solving ability. Neither architecture reaches even 4% held-out once the shared direction is removed. The convolutional inductive bias (spatial weight sharing) may still matter, but this baseline overstated it, and a fair architecture comparison should be run under `--random-goals` or full `--procedural`, not the original fixed-corner setup.

The "random starts alone" claim also does not hold up as a free improvement at this episode budget: it does not beat the fixed-corner baseline for the convolutional model (2/54 vs 12/54) and is roughly a wash for the MLP (2/54 vs 2/54). It is, however, clearly better than fully procedural size+layout+goal randomization (2/54 vs 0-1/54), consistent with the diagnosis that varying every axis at once compounds the sample-efficiency cost.

**Follow-up: is 5,000 episodes just not enough for this harder distribution?** Running `--random-goals` at 20,000 episodes (4x budget, same seeds) answers this directly: train-set fit and pool fit both improved substantially for both architectures (conv train-fit 2.1% to 12.5%, layout 18.8% to 27.1%), but held-out success did not move at all — 2/54 to 2/54 for both. More budget makes the network measurably better at the distribution it's trained on; that improvement does not transfer to the fixed corner-to-corner benchmark. This rules out pure underfitting as the explanation and reinforces the distribution-mismatch diagnosis instead: the fix is a training distribution that doesn't under-sample long routes, not simply more of the same training.

**Fixing the mismatch directly: `--min-separation N`.** This adds a minimum-Manhattan-distance filter to the random start/goal sampling above, biasing it toward routes closer to the training mazes' own corner-to-corner distance (14, for the 10x10 training set) instead of the short hops uniform sampling tends to produce.

```powershell
.\maze_rl.exe --generalization --episodes 5000 --seeds 3 --seed 1 --random-goals --min-separation 10 --csv generalization_random_goals_sep10.csv
```

At the *same* 5,000-episode budget as the original negative result (a quarter of the 20,000-episode budget test above), `--min-separation 10` improved every metric — but the first pass at this used only 3 seeds, and one seed turned out to be doing most of the work. Re-run at 10 seeds each (1,800 held-out evaluations total per config instead of 540) before trusting it:

| Metric | `--random-goals`, min-sep=0 | `--random-goals`, min-sep=10 |
| --- | ---: | ---: |
| conv train-fit | 12/160 (7.5%) | 41/160 (25.6%) |
| layout train-fit | 28/160 (17.5%) | 82/160 (51.3%) |
| conv pool-fit | 672/2560 (26.3%) | 695/2560 (27.1%) |
| layout pool-fit | 903/2560 (35.3%) | 1110/2560 (43.4%) |
| conv all-unseen | 7/180 (3.9%) | 9/180 (5.0%) |
| layout all-unseen | 7/180 (3.9%) | 18/180 (10.0%) |

At 10 seeds, the two architectures start **identical** on the unbiased baseline (3.9% each) — the earlier 3-seed run's "layout already ahead of conv" reading doesn't hold up. The separation fix still helps both, but unevenly: the MLP's held-out score roughly doubles from a real signal spread across most seeds (successes in 8 of 10 seeds, not one outlier), while the convolutional model's improvement is smaller and closer to the edge of what 10 seeds can distinguish from noise. The originally reported 13.0% for the MLP was inflated by a single seed and should be read as 10.0%.

The training-fit numbers are the more interesting thread: the MLP fits its own training distribution roughly 2x better than the convolutional model in *both* conditions (17.5% vs 7.5% unbiased; 51.3% vs 25.6% biased), not just the biased one. That consistency across two different training distributions points at something structural rather than a fluke of one run.

### Isolating capacity from inductive bias: `--wide-conv`

The gap above has two competing explanations: the convolutional model's spatial weight-sharing might be a genuinely worse fit for a task that needs many different goal directions, or it might simply have 38% fewer parameters (13,624 vs the MLP's 22,084) and less raw capacity. Those need different fixes, so `--wide-conv` adds a parameter-matched control: identical architecture (same kernel, stride, conv1 width, dense width), but a wider second convolutional layer (13 filters instead of 8) bringing total parameters to 21,809 — matched to the MLP's 22,084 within 1.2%, and still slightly fewer.

```powershell
.\maze_rl.exe --generalization --episodes 5000 --seeds 10 --seed 1 --random-goals --min-separation 10 --wide-conv --csv generalization_random_goals_sep10_wideconv_10seed.csv
```

| Metric | narrow conv (13,624p) | wide conv (21,809p) | layout MLP (22,084p) |
| --- | ---: | ---: | ---: |
| train-fit | 25.6% | 45.0% | 51.3% |
| pool-fit | 27.1% | 41.9% | 43.4% |
| held-out (all-unseen) | 5.0% | 5.6% | 10.0% |

This cleanly separates two things that looked entangled before. **Capacity explains the train-fit and pool-fit gap almost entirely** — matching parameter count closed most of the distance to the MLP on both metrics (45.0% vs 51.3%; 41.9% vs 43.4%), with nothing else about the architecture changed. **Capacity does not explain the held-out gap** — the wide conv's held-out score (5.6%) is statistically indistinguishable from the narrow conv's (5.0%, per-seed spread checked: 0,2,1,1,0,2,0,0,4,0 — not one outlier), and both remain at roughly half the MLP's 10.0%.

So: giving the convolutional model enough capacity to fit its own training distribution nearly as well as the MLP does *not* give it the MLP's transfer advantage. Whatever is actually driving the held-out gap survives a capacity fix, which points back toward something about the architecture's design (the spatial compression before goal-direction information is mixed in) rather than simple undercapacity — though this isn't fully proven either; a wider net still than 21,809 params, or more training specifically for the wide-conv variant, hasn't been tried, so a residual capacity effect on the held-out number specifically can't be ruled out yet.

See `EXPERIMENTS.md` and `lessons_learned.md` for the full breakdown and the reasoning behind it.

### Sweeping `--min-separation` toward the maximum

`--min-separation 10` only tested one point between the unbiased baseline (0) and the training mazes' own corner-to-corner distance (14). Sweeping 12 and 14 (10 seeds each, standard architectures) shows the relationship is **not monotonic**:

| `--min-separation` | conv unseen | layout unseen | conv train-fit | layout train-fit |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.9% | 3.9% | 7.5% | 17.5% |
| 10 | 5.0% | 10.0% | 25.6% | 51.3% |
| 12 | 2.2% | 7.8% | 41.9% | 66.9% |
| 14 | 12.8% | 11.7% | 95.6% | 100% |

Both architectures dip at 12 before recovering at 14 (per-seed spread checked for both — not an outlier). There's a clean geometric reason for the jump at 14: on this maze's 8x8 interior (coordinates 1-8), the maximum possible Manhattan distance between two cells is exactly 14, achieved by only **2 distinct cell pairs** (the two diagonals) — verified directly, not estimated. `--min-separation 14` therefore doesn't sample "very long routes" so much as collapse training down to (at most) two near-fixed start/goal pairs per maze, which is why train-fit jumps to 95.6-100%: it's a small, highly repeatable set of tasks, similar in spirit to the original fixed-corner baseline. `--min-separation 12`, by contrast, still permits many distinct cell pairs (long but not fully extreme), so it gets the *difficulty* of a demanding distribution without the *repeatability* that either a shorter separation (10) or the fully-collapsed extreme (14) provides — plausibly why it's the worst point of the four.

One thing this rules out: `--min-separation 14` is not simply "put back what we removed." Randomizing which of the two diagonal cells is the start versus the goal means roughly half of training now runs in the *reverse* direction from the held-out suite's canonical `(1,1)` to `(8,8)` — so even at the geometric extreme, conv's held-out score (12.8%) stays well below its original confounded baseline (22.2%). The direction-reversal side effect of `--random-goals`'s symmetric sampling, not just the separation distance, is part of what's keeping these numbers below the original number.

### Render the convolutional learning video

https://github.com/user-attachments/assets/d2289297-ace0-47bc-bece-839ec993c1c7

## PPO on the single maze

A third learner, implemented from scratch in the same dependency-free style: Proximal Policy Optimization, an on-policy policy-gradient method rather than an off-policy value method. Where DQN learns `Q(s,a)` and acts greedily with an epsilon schedule bolted on for exploration, PPO learns a stochastic policy `pi(a|s)` directly (exploration comes from *sampling* that policy) plus a separate value function used only to compute advantages.

It lives in `src/ppo.c` with its own networks, rollout buffer and training loop, rather than implementing `learner.h` — that interface is built around `selectAction(state, epsilon, rng)` and `getQValues`, which don't describe an on-policy stochastic method. This follows the same precedent as `generalization.c`, which is also self-contained.

```powershell
.\maze_rl.exe --ppo --steps 60000 --seeds 5 --seed 1 --compare-dqn --csv ppo.csv
```

or with GNU Make: `make ppo`. `--compare-dqn` additionally trains a DQN baseline on the same maze with the same environment-step budget and evaluation cadence, writing both to one CSV.

The actor is deliberately the same shape as the single-maze DQN (one-hot state, one 64-unit hidden layer, 6,724 parameters) so this isn't secretly a network-size comparison; the critic is the same trunk with a single scalar output (6,529 parameters). Kept as two separate networks rather than a shared trunk with two heads, because two independent backward passes are each verifiable by finite differences.

### Result: both solve it, with opposite strengths

| Algorithm | Optimal route | First greedy solve (env steps) | Wall clock / 60k steps |
| --- | --- | ---: | ---: |
| PPO | 5/5 seeds, 14 steps | 13,107 mean (8.2k-18.4k) | ~120 ms |
| DQN | 5/5 seeds, 14 steps | 4,555 mean (4.1k-6.2k) | ~990 ms |

Both find the same optimal 14-step route on every seed. **DQN is ~2.9x more sample-efficient** (fewer environment steps to first solve, on 5 of 5 seeds) — the expected direction, since replay lets it reuse each transition many times while PPO discards each batch after 4 epochs. **PPO is ~8x faster in wall clock** over the same budget, because DQN runs a 32-sample update every 4 environment steps (15,000 updates) while PPO runs 3,840 larger updates.

Comparison is on **environment steps, not episodes** — an episode means different amounts of experience to each algorithm, and one PPO update (a whole rollout, reused over 4 epochs) isn't comparable to one DQN update (a replay minibatch). "First greedy solve" is checkpointed every 2,048 steps for both, so it's a coarse measure; the ~3x gap far exceeds that granularity and holds on every seed, but the exact multiplier shouldn't be read too precisely.

### PPO on the held-out generalization suite

`--ppo --generalize` runs the same PPO on the 34-maze generalization suite instead of the single maze, with the same `--random-goals` / `--min-separation` options the DQN experiments use.

```powershell
.\maze_rl.exe --ppo --generalize --random-goals --min-separation 10 --steps 575000 --seeds 10 --seed 1 --csv ppo_generalization_sep10.csv
```

Both algorithms share one implementation of the maze suite, the layout-aware encoding, and the step function (exposed from `generalization.c` through `generalization.h`) rather than each keeping a copy — a duplicated encoding that drifted even slightly would silently invalidate the comparison. PPO's actor is 22,084 parameters, matching the layout-aware DQN exactly. Budgets are matched on environment steps: the layout-aware DQN uses 574,926 steps for its 5,000 episodes at this setting, so PPO gets 575,000.

Results over **20 seeds** (360 held-out evaluations per algorithm):

| Algorithm | Parameters | Train-fit | Held-out |
| --- | ---: | ---: | ---: |
| DQN layout MLP | 22,084 | 49.1% | 8.9% (32/360) |
| DQN conv (10 seeds) | 13,624 | 25.6% | 5.0% (9/180) |
| DQN wide conv (10 seeds) | 21,809 | 45.0% | 5.6% (10/180) |
| **PPO layout** | **22,084** | **49.4%** | **18.3% (66/360)** |

**PPO roughly doubles DQN's held-out success at essentially identical train-fit** (49.4% vs 49.1%) — so this is not a difference in how well each fits its training data, it's a difference in what transfers. It comes from changing the *algorithm* while holding architecture, observation, environment, distribution and step budget fixed.

The evidence is solid at this sample size: per-seed head-to-head is **PPO 12 wins, DQN 4, 4 ties**; median 3.5 vs 1.5; a paired test across seeds gives **t = 3.41 (df = 19), p ≈ 0.003**; and the gap survives dropping each side's best seed (17.0% vs 7.6%). Notably the effect got *stronger* going from 10 to 20 seeds (t rose from 1.93 to 3.41), which is what a real effect does and noise generally doesn't.

PPO also reproduces the shortcut finding independently: its train-fit falls from 95.0% on fixed corners to 46.2% with random goals, while held-out stays flat or slightly rises (16.7% to 18.3%) — the same pattern the DQN experiments found, in a different algorithm. (The fixed-corner PPO row is *not* budget-matched — the fixed-corner DQN run uses only 232,395 steps — so it's recorded for completeness, not comparison.)

See `ppo_design.md` for the staging, self-tests and open questions.

### PPO ablations: what each component actually does

The [PPO essentials](#ppo-essentials) section claims each piece of machinery solves a particular problem. These ablations test those claims by removing or varying each one. Hyperparameters are exposed as flags (`--epochs`, `--clip-eps`, `--gae-lambda`, `--entropy-coef`, `--lr`) and a run whose settings differ from the defaults labels itself in the CSV.

**Data reuse (`--epochs K`) is where the sample efficiency comes from.** Single maze, 10 seeds:

| K | Steps to first solve | clip_fraction | Wall clock |
| ---: | ---: | ---: | ---: |
| 1 (no reuse) | 44,442 | 0.000 | 39 ms |
| 4 (default) | 13,312 | 0.012 | 119 ms |
| 10 | 7,578 | 0.039 | 282 ms |
| 20 | 6,963 | 0.061 | 550 ms |

K=1 is essentially vanilla policy gradient and needs **6.4x the environment steps** of K=20. Returns diminish sharply past K=10 (8% better for 2x the compute). `clip_fraction` is *exactly* 0.000 at K=1, as it must be — on the only pass the ratio is identically 1, so clipping cannot bind. A free correctness check.

**Clipping does nothing until updates are aggressive — then it prevents total collapse.** Disabling it (`--clip-eps 1000`) at the default learning rate changes nothing at any K. Sweeping the learning rate at K=10, measuring whether the *final* policy still solves the maze:

| Learning rate | Clipping on | Clipping off | Final entropy (off) |
| ---: | ---: | ---: | ---: |
| 0.0003 (default) | 10/10 | 10/10 | 0.008 |
| 0.003 | 9/10 | **2/10** | 0.024 |
| 0.01 | 7/10 | **0/10** | **0.000** |

At lr=0.01 without clipping, final entropy is exactly 0.000 on all ten seeds — complete policy collapse. Seven of those seeds *did* find the goal early and then destroyed their own policy: the unclipped objective kept raising the winning action's probability during batch reuse until nothing else could be sampled, and a deterministic policy on a wrong action cannot explore back out. This is precisely the failure the clipped objective exists to prevent.

**GAE `lambda` must be high here — and the single-maze answer is the opposite of the real one.** On one fixed maze, low lambda wins monotonically across eight values (λ=0 solves in 10,240 steps vs λ=1.0's 15,974; slower on 9/10 seeds, paired t=3.63). On the generalization suite that reverses completely:

| lambda | Train-fit | Held-out |
| ---: | ---: | ---: |
| 0 | 0.6% | **0.0%** |
| 0.5 | 21.9% | 10.6% |
| 0.95 (default) | 46.2% | 18.3% |
| 1.0 | 37.5% | 22.8% |

λ=0 — the *best* single-maze setting — learns essentially nothing on the real task. Low lambda means trusting the critic; on one fixed maze the critic can be accurate, but across 16 mazes with randomized goals it is badly wrong early, and λ=0 leans entirely on it. (λ=1.0 over the 0.95 default is **not** established: t=0.68. Leave the default alone.)

**The entropy bonus is load-bearing because exploration is the bottleneck.** Generalization suite, 10 seeds:

| c_ent | Train-fit | Held-out | Seeds learning nothing |
| ---: | ---: | ---: | ---: |
| 0 | 20.0% | 8.9% | **5 of 10** |
| 0.01 (default) | 46.2% | 18.3% | 1 of 10 |
| 0.05 | 61.9% | 20.0% | **0 of 10** |

Without the bonus, half the seeds finish having learned nothing (0/16 train-fit; paired t=5.09 for 0 vs 0.05). With goals randomized at separation ≥10 the goal is far away, so a policy that collapses early stops finding it, never receives the +100, and has nothing to learn from. This also explains why *more* exploration pressure **raises** training fit rather than trading against it.

### A caveat that applies to every number above

Every result in this README was scored against the **same 18 held-out mazes**, generated from one hardcoded suite seed. No gradient ever touched them, but roughly 20 configurations have now been compared on them and the best kept — which makes the suite a validation set in practice, and any selected winner optimistically biased.

Measured from the observed per-seed spread, a 10-seed held-out mean carries **±4.1 percentage points** of standard error, and picking the best of 20 equally-good configurations inflates the winner by **~7.7 pp** on average.

- **Robust to this:** PPO vs DQN (9.4 pp, a single pre-specified comparison at 20 seeds, p≈0.003, strengthening as seeds were added), and the large qualitative results — λ=0 failing outright, clipping preventing collapse, half the seeds learning nothing without entropy.
- **Not robust:** any "new best" worth a few points that was *selected* from a sweep. λ=1.0's 22.8% is exactly the shape of number this manufactures for free, which is why it is not claimed over the default.

The suite seed has also never been varied, so strictly these are statements about *these 34 mazes*. The fix — re-running the headline comparisons on several freshly generated suites — is the highest-value outstanding experiment in the project and has not been done.

## Tests

```powershell
make test
```

Everything is checked in one pass, including the hand-written backpropagation.

Shared and DQN-side coverage: environment transitions, replay wraparound, terminal targets, target-network copying, deterministic tabular training, parameter counts, procedural maze solvability, and end-to-end learning of the known optimal 14-action route by both the tabular and DQN agents. Finite-difference gradient checks cover the convolutional and wide-convolutional DQN models.

PPO-side coverage, since hand-rolled policy gradients are easy to get subtly wrong:

- finite-difference gradient checks for the actor and the critic, on both the one-hot and the dense 340-input networks, probing an output-layer *and* a hidden-layer weight each so an error in either backprop stage is caught
- GAE reduces to the one-step TD residual at `lambda = 0` and to the discounted return at `lambda = 1`, and does not chain advantage across an episode boundary
- softmax/log-prob consistency, including stability with logits large enough to overflow a naive `exp`, and uniform entropy equal to `log(4)`
- the clipped branch contributes exactly zero policy gradient while the unclipped branch does not
- determinism: the same seed produces bit-identical networks after a full collect/GAE/train cycle
- parameter counts asserted against their DQN counterparts (6,724 and 22,084)

## Source layout

```text
src/
|-- main.c               Interactive UI and command-line dispatch
|-- environment.c/.h     Shared rewards and transitions
|-- learner.c/.h         Common learner interface (tabular and DQN)
|-- tabular.c/.h         Tabular Q-learning
|-- dqn.c/.h             Neural network, replay, target network, and Adam
|-- ppo.c/.h             PPO: actor/critic, rollouts, GAE, clipped objective
|-- trainer.c/.h         Shared episode and metric logic
|-- benchmark.c/.h       Seeded headless comparison and CSV export
|-- generalization.c/.h  Multi-maze held-out experiment; also exports the
|                          shared maze suite, layout encoding and step
|                          function that PPO evaluates against
|-- rng.c/.h             Deterministic project RNG
|-- maze.c/.h            Maze state and rendering
`-- agent.c/.h           Movement and model-independent visualization
```

`ppo.c` is self-contained rather than implementing `learner.h`, for the reasons in [`ppo_design.md`](ppo_design.md). It shares the *environment* with the DQN experiments (through `generalization.h`) but not the learner interface — so both algorithms provably see identical mazes, observations and dynamics, while each keeps the training loop its own paradigm needs.
