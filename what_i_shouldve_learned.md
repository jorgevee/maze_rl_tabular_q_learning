# What I should have learned from the maze RL project

## State flattening

A grid position can be converted into one integer state by laying every row end to end:

```text
state = y * GRID_WIDTH + x
```

For the start position `(1, 1)` in the 10-column maze:

```text
state = 1 * 10 + 1 = 11
```

The reverse conversion uses integer division and remainder:

```c
Position StateToPosition(int state)
{
    Position position;
    position.x = state % GRID_WIDTH;
    position.y = state / GRID_WIDTH;
    return position;
}
```

This makes a grid position usable as a Q-table row:

```c
qTable[state][action]
```

An important qualification: the flattened number is an identifier, not a meaningful continuous measurement. State `88` is not conceptually twice state `44`, and numerically adjacent states may be separated by a wall.

## The environment owns the rules

`EnvironmentStep()` answers one question:

> If the agent is in this state and chooses this action, what happens next?

It returns:

```c
typedef struct {
    int nextState;
    float reward;
    bool done;
} StepResult;
```

Examples from this maze are:

```text
move to an empty cell:  next state, -1,   not done
move into a wall:       same state, -5,   not done
move into the goal:     goal state, +100, done
```

The learning algorithm should not contain maze rules. It chooses an action; the environment determines the result. Keeping `EnvironmentStep()` independent from rendering lets keyboard control, fast headless training, DQN, tabular Q-learning, and frozen evaluation all use the same world.

## Both learners consume the same transition

The useful common unit is:

```c
typedef struct {
    int state;
    Action action;
    float reward;
    int nextState;
    bool done;
} Transition;
```

Tabular Q-learning stores values directly. DQN predicts them with a neural network. Everything else can be shared:

- environment and rewards
- epsilon-greedy behavior
- episode loop and step limit
- metrics and evaluation
- policy arrows and heatmap rendering

This is also a software-design lesson: define an interface around the behavior that varies. Each learner can reset, select an action, observe a transition, and return four Q-values. The trainer does not need to know how those values are represented.

## Exploration versus greedy tie-breaking

Exploration lets an agent try actions that do not currently appear best. Exploitation uses its current estimates:

```text
with probability epsilon: choose any action
otherwise:                choose a highest-value action
```

Random tie-breaking is different from exploration. A greedy policy can still consume randomness when several actions have exactly the same value. This matters for evaluation reproducibility.

## The tabular update

Tabular Q-learning has one value for every state-action pair:

```text
100 states * 4 actions = 400 Q-values
```

Its update is:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
```

Only `Q(s,a)` changes. For a terminal goal transition, the future value must be zero because no action occurs after the episode ends.

## What changes when the table becomes a DQN

DQN replaces explicit table entries with a function:

```text
Q(state, action; weights)
```

This project uses:

```text
100 one-hot inputs -> 64 ReLU hidden units -> 4 action values
```

The parameter count is:

```text
first-layer weights: 100 * 64 = 6,400
first-layer biases:             64
output weights:       64 * 4 = 256
output biases:                   4
total:                        6,724
```

Changing neural-network weights can change estimates for more than one state. That sharing is the possible benefit of function approximation, but it also makes learning indirect and less stable than changing one table cell.

## Why the DQN input is one-hot

Feeding the scalar flattened state number into a network would invent a numeric ordering that the maze does not possess. A one-hot input treats each number only as an identity:

```text
state 3 -> [0, 0, 0, 1, 0, ...]
state 8 -> [0, 0, 0, 0, 0, ..., 1, ...]
```

This gives DQN the same information as the Q-table and makes the comparison cleaner. It also means the network is mostly learning an expensive version of a lookup table. This experiment does not demonstrate generalization to a new maze.

One-hot inputs also allow a useful C optimization. Only one first-layer input is nonzero, so the forward pass can directly read the active state's weight column. Only that column receives a first-layer gradient for the sample.

## Why replay memory is used

Consecutive maze transitions are correlated: the next sample generally starts where the previous one ended. Training only on that stream can make a network chase its most recent experiences.

The replay buffer stores transitions and samples random minibatches:

```text
capacity: 10,000 transitions
warm-up:     500 transitions
batch size:   32 transitions
```

Replay mixes states and episodes while reusing old experience. The buffer is circular, so new entries overwrite the oldest entries after it fills. Wraparound deserves a direct test because an indexing error can silently train on invalid data.

## Why a target network is used

For a nonterminal transition, the DQN target is:

```text
target = reward + gamma * max(targetNetwork(nextState))
```

Using the rapidly changing online network for both the prediction and target creates a moving-target problem. This implementation holds a separate target network and copies the online parameters into it every 250 optimizer updates.

For a terminal transition:

```text
target = reward
```

Bootstrapping from a terminal goal would teach the agent to include imaginary value after the episode has ended.

## Backpropagation, Huber loss, and Adam

DQN computes an error for the selected action, then uses backpropagation to determine how the contributing weights should change. The error flows backward only through the selected output and hidden ReLU units that were active.

Huber loss behaves like squared error for small mistakes and absolute error for large mistakes. It reduces the influence of unusually large temporal-difference errors, including those caused by the `+100` goal reward early in training.

Adam stores a first and second moment for every parameter. Bias correction is important early because both moment arrays begin at zero. Gradient-norm clipping at `10` adds protection from unusually large minibatch updates.

Implementing this manually in C exposes training costs that a framework normally hides:

- online-network parameters
- a full target-network copy
- two Adam moment arrays
- replay memory
- temporary activations and gradients

Trainable parameter count is therefore not the same as total training memory.

## A fair comparison needs controlled conditions

Both agents use the same:

- maze, start, and goal
- transitions and rewards
- discount factor of `0.95`
- 200-step episode limit
- epsilon schedule from `1.0` to `0.05`
- 5,000 training episodes

Algorithm-specific settings are still necessary. Tabular Q-learning uses `alpha = 0.1`; DQN uses Adam at `0.001`, minibatches, replay configuration, and a target-sync interval. Giving both algorithms the same numeric learning rate would look symmetric but would not be meaningful because the update mechanisms differ.

## Evaluation must be isolated from training

A frozen greedy evaluation should never:

- add replay transitions
- change Q-values or neural-network weights
- decay epsilon
- alter the training random-number sequence

The last item is subtle. An `epsilon = 0` policy can still consume randomness when values tie. Evaluation therefore has its own seeded RNG. Adding or removing a checkpoint cannot change the subsequent training trajectory.

## Training and evaluation metrics answer different questions

Rolling training metrics include epsilon-greedy exploration. Even a trained agent sometimes chooses a bad action while epsilon remains at `0.05`, so its average training route can remain longer than optimal.

Frozen greedy evaluation asks:

> What route does the learned policy choose when it is not deliberately exploring?

The optimal route takes 14 actions and earns `87`:

```text
13 ordinary transitions * -1 = -13
goal transition                = +100
total                          = 87
```

Both metric types should be reported so exploration noise is not mistaken for a bad learned policy.

## What the 10-seed benchmark showed

After 5,000 episodes, every tested seed produced the optimal greedy route:

| Agent | Optimal runs | Greedy route | Parameters | Learner memory | Mean CPU time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tabular | 10 / 10 | 14 steps | 400 | 1,608 bytes | 1.6 ms/run |
| DQN | 10 / 10 | 14 steps | 6,724 | 307,628 bytes | 2,900.3 ms/run |

Both agents were solved by the first 100-episode checkpoint in all ten seeds. Because checkpoints were only recorded every 100 episodes, this does not prove that they learned on the same exact episode.

The lesson is not that DQN is bad. The lesson is to match the representation to the problem. On a tiny, fixed, discrete state space, tabular Q-learning is simpler, faster, smaller, easier to inspect, and stable. DQN becomes attractive when a table is impractical or when useful structure can be shared across a much larger state space.

## Reproducibility is part of correctness

The original learner used raylib randomness. That was convenient for visualization, but coupled learning to the graphics library and made headless experiments harder to reproduce.

The deterministic project RNG now controls:

- epsilon exploration
- greedy tie-breaking
- DQN initialization
- replay sampling

Two identical benchmark runs produced identical metrics after excluding elapsed time. Determinism makes subtle regressions in indexing, reset behavior, or RNG use easier to find.

## Tests that mattered

End-to-end success alone is insufficient because a small maze may occasionally be solved even with a defect. The test suite checks:

- normal moves, wall collisions, and goal transitions
- replay insertion and wraparound
- terminal targets
- target-network copying
- parameter counts
- same-seed determinism
- a finite-difference gradient check
- both learners reaching the known 14-step optimum

The finite-difference check perturbs one weight, measures the numerical change in loss, and compares that slope with the analytic backpropagation gradient. This is particularly valuable when implementing neural-network math manually.

## Concerns and limitations

- This is one fixed deterministic maze. It does not demonstrate DQN generalization.
- One-hot states let the network memorize state identities; a new-maze experiment needs inputs that describe maze structure.
- Ten successful seeds are encouraging, not proof of universal stability.
- The 100-episode checkpoint interval is too coarse to compare exact learning speed early in training.
- CPU timings depend on the machine and compiler. Tabular runs are so short that timer resolution affects their average.
- Reported learner memory covers persistent learner state, not raylib, executable code, allocator overhead, or stack temporaries.
- A shared epsilon schedule is easy to compare but may not be optimal for either algorithm individually.
- DQN training is synchronous with the raylib loop, so very high training speeds can reduce UI responsiveness.
- Resetting DQN produces fresh random weights. For an exact rerun, the RNG seed must also be restored.

## Good next experiments

1. Train across multiple maze layouts and evaluate on held-out mazes.
2. Replace one-hot identity with inputs that describe local or complete maze geometry.
3. Record checkpoints more frequently during the first 100 episodes.
4. Remove replay or the target network one at a time and observe the stability difference.
5. Add Double DQN and compare its target calculation with ordinary DQN.
6. Move long training work off the render loop if UI responsiveness becomes important.

The important question for future DQN work is not only, "Can it solve the maze?" It is:

> Does the learned function reuse structure in a way that a lookup table cannot?
