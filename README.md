# Q-Learning Maze

A small C and raylib project for learning tabular Q-learning. The program renders a fixed 10 by 10 maze, supports keyboard movement, and contains the core pieces needed to train an agent.

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
- A training-episode loop with a 200-step limit

The next task is to connect `TrainEpisode()` to a key in `main.c`, run an episode, and then add repeated training and epsilon decay.

## Build and run

From PowerShell:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + $env:PATH
gcc -Wall -Wextra -pedantic src\main.c src\maze.c src\agent.c src\environment.c -o main.exe -Ic:\msys64\mingw64\include -Lc:\msys64\mingw64\lib -lraylib -lopengl32 -lgdi32 -lwinmm -lm
.\main.exe
```

Use the arrow keys to move the red agent. Walls block movement, the blue cell is the start, and the green cell is the goal.

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

More detailed learning notes are in [`what_i_shouldve_learned.md`](what_i_shouldve_learned.md), and the remaining roadmap is in [`plan.md`](plan.md).
