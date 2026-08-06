# Q-Learning Maze

A small C and raylib project for learning tabular Q-learning. The program renders a fixed 10 by 10 maze, supports keyboard movement, and contains the core pieces needed to train an agent.

![Trained Q-learning maze showing policy arrows and the Q-value heatmap](assets/mazesc1.png)

*The trained policy and value heatmap after the agent reaches the green goal cell.*

## Current progress

Implemented:

- raylib window and maze rendering
- Human movement with wall collision
- Position-to-state flattening (`state = y * width + x`)
- Environment transitions with rewards and goal termination
- A 100-state by 4-action Q-table
- Greedy selection with random tie-breaking
- Epsilon-greedy exploration
- The Q-learning/Bellman update
- Batched headless training with pause/resume and a 200-step episode limit
- Epsilon decay from `1.0` to a minimum of `0.05`
- A verified 5,000-episode run that learned the optimal 14-action route
- Rolling 100-episode success rate and average successful path length
- A toggleable learned-policy view with stable directional arrows
- A Q-value heatmap that highlights high-value routes and avoided cells
- Animated greedy checkpoint previews during otherwise headless training

Most steps in the project plan are complete. The next task is to add the remaining interactive controls and a greedy policy demonstration.

## Build and run

From PowerShell:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + $env:PATH
gcc -Wall -Wextra -pedantic src\main.c src\maze.c src\agent.c src\environment.c -o main.exe -Ic:\msys64\mingw64\include -Lc:\msys64\mingw64\lib -lraylib -lopengl32 -lgdi32 -lwinmm -lm
.\main.exe
```

## Controls

| Input | Action |
| --- | --- |
| Arrow keys | Move the red human-controlled agent |
| `T` | Start or pause batched Q-learning training |
| `P` | Show or hide learned-policy arrows |
| `V` | Show or hide the learned Q-value heatmap |

Walls block movement, the blue cell is the start, and the green cell is the goal. Episode count, epsilon, and training status are displayed in the window.

## Verified training result

A 5,000-episode run completed successfully:

| Episode | Success rate (last 100) | Average successful steps | Epsilon |
| ---: | ---: | ---: | ---: |
| 100 | 85% | 66.48 | 0.606 |
| 200 | 100% | 26.92 | 0.367 |
| 600 | 100% | 14.94 | 0.050 |
| 5,000 | 100% | 14.62 | 0.050 |

The frequently observed reward of `87` represents an optimal 14-action route: 13 ordinary transitions at `-1`, followed by the `+100` goal transition. The final average remains slightly above 14 because training retains 5% exploration.

Training also pauses to animate frozen greedy-policy previews at episodes 0, 100, 500, 1,000, and 5,000. The untrained episode-0 policy hit the 60-step preview limit; every checkpoint from episode 100 onward reached the goal in the optimal 14 steps. This highlights the difference between noisy epsilon-greedy training metrics and evaluation with `epsilon = 0`.

## Source layout

```text
src/
|-- main.c          Window loop, input, and rendering calls
|-- maze.c/.h       Grid, drawing, and state conversion
|-- agent.c/.h      Movement, Q-table, action selection, and learning
`-- environment.c/.h  Rewards, transitions, and episode step limit
```

## RL notes

Random tie-breaking and exploration are different. Greedy selection chooses only actions with the highest Q-value and randomly resolves equal values. Epsilon-greedy selection sometimes ignores the table and chooses any action; that random choice may still happen to be the greedy action.

Each experience is a transition:

```text
(state, action, reward, next state)
```

The update is:

```text
Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(next state) - Q(s,a)]
```

`alpha` controls how far the current estimate moves toward the new target. `gamma` controls how much future rewards matter. Only `qTable[state][action]` changes during an update.

For a terminal goal transition, future Q-value is zero. The agent must not learn from imaginary actions after the episode has ended. An episode also stops after 200 steps so early random behavior cannot continue forever.