# Tabular Q-Learning vs DQN Maze

A native C and raylib teaching project that solves the same fixed 10 by 10 maze with two reinforcement-learning agents:

- tabular Q-learning with a 100 by 4 Q-table
- a dependency-free Deep Q-Network (DQN) implemented from scratch in C

![Trained Q-learning maze showing policy arrows and the Q-value heatmap](assets/mazesc1.png)

The comparison is deliberately unfair to DQN in an instructive way. A Q-table is an excellent fit for this tiny, fully observable discrete environment. DQN can learn the same optimal policy, but needs a neural network, replay memory, minibatch optimization, and a target network to do it.

## Build and run

The Makefile expects the MSYS2 raylib installation at `C:/msys64/mingw64`. If GNU Make is installed:

```powershell
make
.\maze_rl.exe
```

The executable supports both the interactive raylib application and headless experiments. No external machine-learning library is required.

On the current Windows setup, where `gcc` is available but `make` is not on `PATH`, use:

```powershell
gcc -std=c11 -O2 -Wall -Wextra -pedantic src\main.c src\maze.c src\agent.c src\environment.c src\rl.c src\rng.c src\learner.c src\tabular.c src\dqn.c src\trainer.c src\benchmark.c src\generalization.c -o maze_rl.exe -IC:\msys64\mingw64\include -LC:\msys64\mingw64\lib -lraylib -lopengl32 -lgdi32 -lwinmm -lm
.\maze_rl.exe
```

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

See `EXPERIMENTS.md` and `lessons_learned.md` for the full breakdown and the reasoning behind it.

### Render the convolutional learning video

https://github.com/user-attachments/assets/d2289297-ace0-47bc-bece-839ec993c1c7

## Algorithms

The complete training flow is available as a Graphviz diagram in [`assets/dqn_algorithm.dot`](assets/dqn_algorithm.dot). Render it after installing Graphviz with:

```powershell
dot -Tsvg assets/dqn_algorithm.dot -o assets/dqn_algorithm.svg
```

Both agents receive exactly the same transition:

```text
(state, action, reward, next state, done)
```

They share the maze, rewards, 200-step episode limit, discount factor (`0.95`), and epsilon schedule (`1.0` decaying to `0.05`).

The tabular update is:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
```

The DQN uses:

- a 100-element one-hot state input
- a `100 -> 64 ReLU -> 4` network (6,724 trainable parameters)
- He weight initialization
- a 10,000-transition circular replay buffer
- 500-transition warm-up and 32-transition minibatches
- Huber loss and Adam with learning rate `0.001`
- gradient-norm clipping at `10`
- a target network copied every 250 optimizer updates

One-hot input makes the baseline honest: it gives DQN the same state identity used by the Q-table without pretending flattened state numbers have meaningful numeric distance.

## Summary of mathematical formalisms

Let `s_t` be the current state, `a_t` one of the four actions in the action set, `theta` the online-network parameters, and `theta_target` (written `theta^-` below) the target-network parameters.

### Epsilon-greedy action selection

Exploration samples an action uniformly; exploitation selects an action with maximum online-network value. The implementation randomly selects among exact maximizing ties:

```text
a_t = uniform random action                          if u < epsilon
a_t = random member of argmax_a Q(s_t, a; theta)      if u >= epsilon

  where u ~ Uniform[0, 1)
```

### Bellman optimality target

For replay transition `j`, let `d_j = 1` indicate termination. The fixed target-network value is:

```text
y_j = r_j                                             if d_j = 1
y_j = r_j + gamma * max_a' Q(s'_j, a'; theta^-)       if d_j = 0

  where gamma = 0.95
```

The terminal branch deliberately contains no future value because no action occurs after reaching the goal.

### Minibatch Huber loss

For a uniformly sampled replay minibatch `B` of size 32, define the temporal-difference error `e_j = y_j - Q(s_j, a_j; theta)`. The implementation uses Huber loss with threshold 1:

```text
Huber_1(e) = 0.5 * e^2      if |e| <= 1
Huber_1(e) = |e| - 0.5      if |e| > 1

L(theta) = (1 / |B|) * sum_over_j_in_B( Huber_1(y_j - Q(s_j, a_j; theta)) )
  where |B| = 32
```

### Gradient clipping and optimization

The minibatch gradient is clipped to Euclidean norm 10, then passed to Adam:

```text
g = gradient of L(theta) with respect to theta

g_clipped = g                       if ||g||_2 <= 10
g_clipped = 10 * g / ||g||_2        if ||g||_2 > 10

theta <- Adam(theta, g_clipped; alpha=1e-3, beta1=0.9, beta2=0.999, epsilon_adam=1e-8)
```

### Periodic target-network copy

The target is held fixed between hard synchronization steps:

```text
theta^- <- theta      every C = 250 optimizer updates
```

### Exploration schedule

After each training episode:

```text
epsilon <- max(epsilon_min, epsilon * lambda_epsilon)

  lambda_epsilon = 0.995
  epsilon_min    = 0.05
  epsilon_0      = 1.0   (starting value)
```

### Network equations and tensor dimensions

For one-hot state vector `x(s)` in `R^100`:

```text
h = ReLU(W1 * x(s) + b1)
Q(s, .; theta) = W2 * h + b2

  W1 in R^(64x100),  b1 in R^64,  W2 in R^(4x64),  b2 in R^4
```

Therefore the online network contains exactly:

```text
(64 * 100) + 64 + (4 * 64) + 4 = 6,724
```

Thus, the online network has **6,724 trainable parameters**.

Weights use He initialization with variance `2 / n_in`:

```text
W_ij ~ Normal(0, 2 / n_in)
```

After the online weights are initialized, the target network begins as an exact copy of the online network:

```text
theta^- <- theta
```

## Tests

```powershell
make test
```

Tests cover environment transitions, replay wraparound, terminal targets, target copying, a finite-difference gradient check, deterministic tabular training, parameter counts, and end-to-end learning of the known optimal 14-action route by both agents.

## Source layout

```text
src/
|-- main.c              Interactive UI and command-line dispatch
|-- environment.c/.h   Shared rewards and transitions
|-- learner.c/.h       Common learner interface
|-- tabular.c/.h       Tabular Q-learning
|-- dqn.c/.h           Neural network, replay, target network, and Adam
|-- trainer.c/.h       Shared episode and metric logic
|-- benchmark.c/.h     Seeded headless comparison and CSV export
|-- generalization.c/.h  Multi-maze and cross-size held-out experiment
|-- rng.c/.h           Deterministic project RNG
|-- maze.c/.h          Maze state and rendering
`-- agent.c/.h         Movement and model-independent visualization
```
