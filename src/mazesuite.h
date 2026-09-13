#ifndef MAZESUITE_H
#define MAZESUITE_H

#include <stdbool.h>
#include <stdint.h>
#include "rl.h"
#include "rng.h"

/* The maze environment the headless experiments run on: suite generation,
   the agent-centered observation, and the step function.

   This lives in its own module because more than one algorithm is evaluated
   on it (DQN in generalization.c, PPO in ppo.c) and a comparison between
   them is only meaningful if they see byte-identical mazes, observations and
   dynamics. One definition, one place. It is also the module a 3D variant
   sits beside, rather than being buried inside a DQN experiment.

   Distinct from maze.h, which is the single fixed maze the interactive
   raylib app renders. */

#define GEN_MAX_SIZE 12
#define GEN_MAX_CELLS (GEN_MAX_SIZE * GEN_MAX_SIZE)
#define GEN_ACTIONS 4
#define GEN_CROP_RADIUS 6
#define GEN_CROP_SIDE (2 * GEN_CROP_RADIUS + 1)
#define GEN_LAYOUT_INPUT (2 * GEN_CROP_SIDE * GEN_CROP_SIDE + 2)
#define GEN_MAX_STEPS 200
#define GEN_TRAIN_MAZES 16
#define GEN_TEST_PER_GROUP 6
#define GEN_MAZE_COUNT (GEN_TRAIN_MAZES + 3 * GEN_TEST_PER_GROUP)
/* The suite seed every published result in this project was generated with. */
#define GEN_SUITE_SEED UINT64_C(20260830)

typedef struct {
    int width;
    int height;
    unsigned char wall[GEN_MAX_CELLS];
    int startState;
    int goalState;
    int optimalSteps;
    uint64_t generationSeed;
    const char *split;
    char name[32];
} ExperimentMaze;

/* Everything needed to observe and step a maze, decoupled from the fixed
   suite so a procedurally generated maze works the same way. */
typedef struct {
    int width;
    int height;
    unsigned char wall[GEN_MAX_CELLS];
    int goalState;
} GenMazeView;

typedef struct {
    int nextState;
    float reward;
    bool done;
} GenStepOutcome;

/* Cell/state conversions. States index a GEN_MAX_SIZE-wide row-major grid so
   that mazes of different sizes share one state numbering. */
int StateOf(int x, int y);
int StateX(int state);
int StateY(int state);

/* Breadth-first shortest path, or -1 when the goal is unreachable. */
int ShortestPathGeneric(
    int width,
    int height,
    const unsigned char wall[GEN_MAX_CELLS],
    int startState,
    int goalState);
int ShortestPath(const ExperimentMaze *maze);

/* The deterministic 34-maze suite: 16 training mazes plus 18 held-out. */
void BuildMazeSuite(ExperimentMaze mazes[GEN_MAZE_COUNT], uint64_t suiteSeed);
GenMazeView ViewOfMaze(const ExperimentMaze *maze);

/* Agent-centered crop: wall and goal planes plus normalized goal
   displacement. Zeroes `output` first. */
void EncodeLayout(const GenMazeView *maze, int state, float output[GEN_LAYOUT_INPUT]);

/* One environment step. +100 at the goal, -5 into a wall, -1 otherwise. */
GenStepOutcome StepMaze(const GenMazeView *maze, int state, Action action);

/* Training-distribution options shared by both algorithms. */
void GenerateProceduralMaze(
    GenMazeView *view,
    int *startState,
    int minSize,
    int maxSize,
    Rng *rng);
void RandomizeStartGoal(
    const ExperimentMaze *maze,
    int *startState,
    int *goalState,
    int minSeparation,
    Rng *rng);

#endif
