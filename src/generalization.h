#ifndef GENERALIZATION_H
#define GENERALIZATION_H

#include <stdbool.h>
#include <stdint.h>
#include "rl.h"
#include "rng.h"

/* Maze-suite geometry and the layout-aware observation shape. These live in
   the header because more than one algorithm is evaluated on this suite:
   a PPO-vs-DQN comparison is only meaningful if both see byte-identical
   mazes, observations and dynamics, so the definitions have exactly one
   home rather than a copy per algorithm. */
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

int RunGeneralizationExperiment(int argc, char **argv);
int RunGeneralizationVideo(int argc, char **argv);
bool GeneralizationRunSelfTests(void);

/* Shared environment surface, so another algorithm can be run on this exact
   suite without duplicating (and eventually diverging from) its definitions. */
void GenBuildMazeSuite(ExperimentMaze mazes[GEN_MAZE_COUNT], uint64_t suiteSeed);
GenMazeView GenViewOfMaze(const ExperimentMaze *maze);
void GenEncodeLayout(const GenMazeView *maze, int state, float output[GEN_LAYOUT_INPUT]);
GenStepOutcome GenStep(const GenMazeView *maze, int state, Action action);
void GenRandomizeStartGoal(
    const ExperimentMaze *maze,
    int *startState,
    int *goalState,
    int minSeparation,
    Rng *rng);

#endif
