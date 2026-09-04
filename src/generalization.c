#include "generalization.h"
#include "raylib.h"
#include "rng.h"
#include "rl.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GEN_MAX_SIZE 12
#define GEN_MAX_CELLS (GEN_MAX_SIZE * GEN_MAX_SIZE)
#define GEN_ACTIONS 4
#define GEN_TRAIN_MAZES 16
#define GEN_TEST_PER_GROUP 6
#define GEN_POOL_CAP 256
#define GEN_MAZE_COUNT (GEN_TRAIN_MAZES + 3 * GEN_TEST_PER_GROUP)
#define GEN_REPLAY_CAPACITY 10000
#define GEN_REPLAY_WARMUP 500
#define GEN_BATCH_SIZE 32
#define GEN_TARGET_SYNC 250
#define GEN_MAX_STEPS 200
#define GEN_CROP_RADIUS 6
#define GEN_CROP_SIDE (2 * GEN_CROP_RADIUS + 1)
#define GEN_LAYOUT_INPUT (2 * GEN_CROP_SIDE * GEN_CROP_SIDE + 2)
#define GEN_POSITION_INPUT GEN_MAX_CELLS
#define GEN_MAX_INPUT GEN_LAYOUT_INPUT
#define GEN_MAX_HIDDEN 64
#define GEN_CONV_KERNEL 3
#define GEN_CONV1_FILTERS 4
#define GEN_CONV1_SIDE (GEN_CROP_SIDE - GEN_CONV_KERNEL + 1)
#define GEN_CONV1_COUNT (GEN_CONV1_FILTERS * GEN_CONV1_SIDE * GEN_CONV1_SIDE)
#define GEN_CONV2_FILTERS 8
#define GEN_CONV2_STRIDE 2
#define GEN_CONV2_SIDE ((GEN_CONV1_SIDE - GEN_CONV_KERNEL) / GEN_CONV2_STRIDE + 1)
#define GEN_CONV2_COUNT (GEN_CONV2_FILTERS * GEN_CONV2_SIDE * GEN_CONV2_SIDE)
#define GEN_CONV_DENSE_INPUT (GEN_CONV2_COUNT + 2)
#define GEN_VIDEO_MAZES 4
#define GEN_VIDEO_MAX_CHECKPOINTS 6
#define GEN_VIDEO_DEFAULT_FPS 30
#define GEN_VIDEO_DEFAULT_WIDTH 1280
#define GEN_VIDEO_DEFAULT_HEIGHT 720
#define GEN_VIDEO_DEFAULT_SECONDS 45

typedef enum {
    OBS_POSITION,
    OBS_LAYOUT,
    OBS_CONV
} ObservationKind;

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

/* Everything Encode()/ForwardState() need to observe a maze, decoupled from
   the fixed ExperimentMaze suite so a freshly generated procedural maze can
   be observed the same way as one of the deterministic held-out mazes. */
typedef struct {
    int width;
    int height;
    unsigned char wall[GEN_MAX_CELLS];
    int goalState;
} GenMazeView;

typedef struct {
    GenMazeView maze;
    int state;
    Action action;
    float reward;
    int nextState;
    bool done;
} GenTransition;

typedef struct {
    GenTransition entries[GEN_REPLAY_CAPACITY];
    int count;
    int next;
} GenReplay;

typedef struct {
    int inputSize;
    int hiddenSize;
    int parameterCount;
    float *online;
    float *target;
    float *firstMoment;
    float *secondMoment;
    float *gradient;
    GenReplay replay;
    int updates;
    int environmentSteps;
    ObservationKind observation;
} GenDqn;

typedef struct {
    float input[GEN_MAX_INPUT];
    float conv1[GEN_CONV1_COUNT];
    float conv2[GEN_CONV2_COUNT];
    float hidden[GEN_MAX_HIDDEN];
    float values[GEN_ACTIONS];
} GenForward;

typedef struct {
    bool reachedGoal;
    int steps;
    float totalReward;
} GenEpisode;

typedef struct {
    int episodes;
    int seeds;
    uint64_t firstSeed;
    const char *csvPath;
    bool procedural;
    int proceduralMinSize;
    int proceduralMaxSize;
    int proceduralRegenEvery;
    bool randomGoals;
    int randomGoalsMinSeparation;
} GenOptions;

typedef struct {
    int episodes;
    uint64_t seed;
    const char *framesPath;
    int fps;
    int width;
    int height;
    int seconds;
} GenVideoOptions;

typedef struct {
    int states[GEN_MAX_STEPS + 1];
    int steps;
    bool reachedGoal;
    float totalReward;
} GenVideoTrajectory;

typedef struct {
    int episode;
    float *parameters;
    GenVideoTrajectory trajectories[GEN_VIDEO_MAZES];
} GenVideoCheckpoint;

static int StateOf(int x, int y) { return y * GEN_MAX_SIZE + x; }
static int StateX(int state) { return state % GEN_MAX_SIZE; }
static int StateY(int state) { return state / GEN_MAX_SIZE; }

static int ShortestPathGeneric(
    int width,
    int height,
    const unsigned char wall[GEN_MAX_CELLS],
    int startState,
    int goalState)
{
    int distance[GEN_MAX_CELLS];
    int queue[GEN_MAX_CELLS];
    for (int i = 0; i < GEN_MAX_CELLS; i++) distance[i] = -1;
    int front = 0;
    int back = 0;
    queue[back++] = startState;
    distance[startState] = 0;
    static const int dx[GEN_ACTIONS] = {0, 1, 0, -1};
    static const int dy[GEN_ACTIONS] = {-1, 0, 1, 0};
    while (front < back) {
        int state = queue[front++];
        if (state == goalState) return distance[state];
        int x = StateX(state);
        int y = StateY(state);
        for (int action = 0; action < GEN_ACTIONS; action++) {
            int nx = x + dx[action];
            int ny = y + dy[action];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            int next = StateOf(nx, ny);
            if (wall[next] || distance[next] >= 0) continue;
            distance[next] = distance[state] + 1;
            queue[back++] = next;
        }
    }
    return -1;
}

static int ShortestPath(const ExperimentMaze *maze)
{
    return ShortestPathGeneric(maze->width, maze->height, maze->wall,
        maze->startState, maze->goalState);
}

static GenMazeView ViewOfMaze(const ExperimentMaze *maze)
{
    GenMazeView view;
    view.width = maze->width;
    view.height = maze->height;
    memcpy(view.wall, maze->wall, sizeof(view.wall));
    view.goalState = maze->goalState;
    return view;
}

static int RandomOpenInteriorCell(int width, int height, Rng *rng)
{
    int x = 1 + RngRange(rng, width - 2);
    int y = 1 + RngRange(rng, height - 2);
    return StateOf(x, y);
}

/* Generates one procedural training maze: a random square size in
   [minSize, maxSize], with a random (not corner-fixed) start and goal cell.
   Mirrors GenerateMaze's rejection-sampling approach (random interior walls
   at 24% density, keep the best-connected attempt out of 4000, require a
   BFS path meaningfully longer than the direct Manhattan distance) but
   randomizes start/goal instead of pinning them to opposite corners. */
static void GenerateProceduralMaze(
    GenMazeView *view,
    int *startState,
    int minSize,
    int maxSize,
    Rng *rng)
{
    int size = minSize + (minSize == maxSize ? 0 : RngRange(rng, maxSize - minSize + 1));
    view->width = size;
    view->height = size;
    unsigned char best[GEN_MAX_CELLS];
    int bestStart = StateOf(1, 1);
    int bestGoal = StateOf(size - 2, size - 2);
    int bestDistance = -1;

    for (int attempt = 0; attempt < 4000; attempt++) {
        memset(view->wall, 1, sizeof(view->wall));
        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++)
                view->wall[StateOf(x, y)] = RngFloat(rng) < 0.24f ? 1 : 0;
        }
        int candidateStart = RandomOpenInteriorCell(size, size, rng);
        int candidateGoal = RandomOpenInteriorCell(size, size, rng);
        if (candidateGoal == candidateStart) continue;
        view->wall[candidateStart] = 0;
        view->wall[candidateGoal] = 0;
        int distance = ShortestPathGeneric(size, size, view->wall, candidateStart, candidateGoal);
        if (distance > bestDistance) {
            bestDistance = distance;
            bestStart = candidateStart;
            bestGoal = candidateGoal;
            memcpy(best, view->wall, sizeof(best));
        }
        int manhattan = abs(StateX(candidateGoal) - StateX(candidateStart)) +
            abs(StateY(candidateGoal) - StateY(candidateStart));
        if (distance >= manhattan + 2) {
            *startState = candidateStart;
            view->goalState = candidateGoal;
            return;
        }
    }

    if (bestDistance < 0) {
        /* Extremely unlikely fallback: no attempt connected start to goal.
           Fall back to a fully open interior so the maze is always solvable. */
        memset(best, 1, sizeof(best));
        for (int y = 1; y < size - 1; y++)
            for (int x = 1; x < size - 1; x++) best[StateOf(x, y)] = 0;
        bestStart = StateOf(1, 1);
        bestGoal = StateOf(size - 2, size - 2);
    }
    memcpy(view->wall, best, sizeof(best));
    *startState = bestStart;
    view->goalState = bestGoal;
}

static int CollectOpenInteriorCells(const ExperimentMaze *maze, int cells[GEN_MAX_CELLS])
{
    int count = 0;
    for (int y = 1; y < maze->height - 1; y++) {
        for (int x = 1; x < maze->width - 1; x++) {
            int state = StateOf(x, y);
            if (!maze->wall[state]) cells[count++] = state;
        }
    }
    return count;
}

/* Isolates the "random starts" claim from procedural generation: keeps a
   fixed maze's wall layout exactly as generated, and only swaps in a random
   pair of open cells as start/goal, subject to the same minimum-distance
   rejection sampling as GenerateProceduralMaze. `minSeparation` additionally
   rejects any candidate pair whose Manhattan distance falls short of it,
   biasing the sampled start/goal toward longer routes so the training
   distribution stops under-sampling the long corner-to-corner case the
   held-out suite always tests (0 disables this and reproduces the original
   uniform-random behavior exactly). Falls back to the maze's own canonical
   start/goal if no qualifying open pair is ever found (should not happen
   for a validated, connected maze at minSeparation 0, but the fixed suite is
   untouched here so this is defensive rather than load-bearing; at a high
   minSeparation on a small maze, this fallback is the expected outcome and
   itself a maximally-separated pair). */
static void RandomizeStartGoal(
    const ExperimentMaze *maze,
    int *startState,
    int *goalState,
    int minSeparation,
    Rng *rng)
{
    int open[GEN_MAX_CELLS];
    int count = CollectOpenInteriorCells(maze, open);
    int bestStart = maze->startState;
    int bestGoal = maze->goalState;
    int bestDistance = -1;
    int attempts = count > 1 ? 200 : 0;
    for (int attempt = 0; attempt < attempts; attempt++) {
        int a = RngRange(rng, count);
        int b = RngRange(rng, count);
        if (a == b) continue;
        int candidateStart = open[a];
        int candidateGoal = open[b];
        int manhattan = abs(StateX(candidateGoal) - StateX(candidateStart)) +
            abs(StateY(candidateGoal) - StateY(candidateStart));
        if (manhattan < minSeparation) continue;
        int distance = ShortestPathGeneric(
            maze->width, maze->height, maze->wall, candidateStart, candidateGoal);
        if (distance > bestDistance) {
            bestDistance = distance;
            bestStart = candidateStart;
            bestGoal = candidateGoal;
        }
        if (distance >= manhattan + 2) {
            *startState = candidateStart;
            *goalState = candidateGoal;
            return;
        }
    }
    *startState = bestStart;
    *goalState = bestGoal;
}

static void GenerateMaze(
    ExperimentMaze *maze,
    int width,
    int height,
    uint64_t seed,
    const char *split,
    const char *name)
{
    Rng rng;
    RngSeed(&rng, seed);
    maze->width = width;
    maze->height = height;
    maze->startState = StateOf(1, 1);
    maze->goalState = StateOf(width - 2, height - 2);
    maze->split = split;
    maze->generationSeed = seed;
    snprintf(maze->name, sizeof(maze->name), "%s", name);
    int directDistance = (width - 3) + (height - 3);
    int bestDistance = -1;
    unsigned char best[GEN_MAX_CELLS] = {0};

    for (int attempt = 0; attempt < 4000; attempt++) {
        memset(maze->wall, 1, sizeof(maze->wall));
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int state = StateOf(x, y);
                maze->wall[state] = RngFloat(&rng) < 0.24f ? 1 : 0;
            }
        }
        maze->wall[maze->startState] = 0;
        maze->wall[maze->goalState] = 0;
        int distance = ShortestPath(maze);
        if (distance > bestDistance) {
            bestDistance = distance;
            memcpy(best, maze->wall, sizeof(best));
        }
        if (distance >= directDistance + 2) {
            maze->optimalSteps = distance;
            return;
        }
    }
    memcpy(maze->wall, best, sizeof(best));
    maze->optimalSteps = bestDistance;
}

static void BuildMazeSuite(ExperimentMaze mazes[GEN_MAZE_COUNT], uint64_t suiteSeed)
{
    int index = 0;
    for (int i = 0; i < GEN_TRAIN_MAZES; i++, index++) {
        char name[32];
        snprintf(name, sizeof(name), "train_10_%02d", i + 1);
        GenerateMaze(&mazes[index], 10, 10, suiteSeed + 100 + (uint64_t)i, "train", name);
    }
    const int sizes[3] = {10, 8, 12};
    const char *splits[3] = {"heldout_same", "heldout_smaller", "heldout_larger"};
    for (int group = 0; group < 3; group++) {
        for (int i = 0; i < GEN_TEST_PER_GROUP; i++, index++) {
            char name[32];
            snprintf(name, sizeof(name), "%s_%02d", splits[group], i + 1);
            GenerateMaze(
                &mazes[index],
                sizes[group],
                sizes[group],
                suiteSeed + 1000 + (uint64_t)(group * 100 + i),
                splits[group],
                name);
        }
    }
}

static int MlpParameterCount(int inputSize, int hiddenSize)
{
    return hiddenSize * inputSize + hiddenSize + GEN_ACTIONS * hiddenSize + GEN_ACTIONS;
}

static int MlpW1Offset(void) { return 0; }
static int MlpB1Offset(const GenDqn *dqn) { return dqn->hiddenSize * dqn->inputSize; }
static int MlpW2Offset(const GenDqn *dqn) { return MlpB1Offset(dqn) + dqn->hiddenSize; }
static int MlpB2Offset(const GenDqn *dqn)
{
    return MlpW2Offset(dqn) + GEN_ACTIONS * dqn->hiddenSize;
}

static int Conv1WeightsOffset(void) { return 0; }
static int Conv1BiasesOffset(void)
{
    return GEN_CONV1_FILTERS * 2 * GEN_CONV_KERNEL * GEN_CONV_KERNEL;
}
static int Conv2WeightsOffset(void) { return Conv1BiasesOffset() + GEN_CONV1_FILTERS; }
static int Conv2BiasesOffset(void)
{
    return Conv2WeightsOffset() +
        GEN_CONV2_FILTERS * GEN_CONV1_FILTERS * GEN_CONV_KERNEL * GEN_CONV_KERNEL;
}
static int ConvDenseWeightsOffset(void) { return Conv2BiasesOffset() + GEN_CONV2_FILTERS; }
static int ConvDenseBiasesOffset(void)
{
    return ConvDenseWeightsOffset() + GEN_MAX_HIDDEN * GEN_CONV_DENSE_INPUT;
}
static int ConvOutputWeightsOffset(void) { return ConvDenseBiasesOffset() + GEN_MAX_HIDDEN; }
static int ConvOutputBiasesOffset(void)
{
    return ConvOutputWeightsOffset() + GEN_ACTIONS * GEN_MAX_HIDDEN;
}
static int ConvParameterCount(void) { return ConvOutputBiasesOffset() + GEN_ACTIONS; }

static void Encode(
    const GenDqn *dqn,
    const GenMazeView *maze,
    int state,
    float output[GEN_MAX_INPUT])
{
    memset(output, 0, sizeof(float) * (size_t)dqn->inputSize);
    if (dqn->observation == OBS_POSITION) {
        output[state] = 1.0f;
        return;
    }

    int agentX = StateX(state);
    int agentY = StateY(state);
    int goalX = StateX(maze->goalState);
    int goalY = StateY(maze->goalState);
    int plane = GEN_CROP_SIDE * GEN_CROP_SIDE;
    for (int relativeY = -GEN_CROP_RADIUS; relativeY <= GEN_CROP_RADIUS; relativeY++) {
        for (int relativeX = -GEN_CROP_RADIUS; relativeX <= GEN_CROP_RADIUS; relativeX++) {
            int cropX = relativeX + GEN_CROP_RADIUS;
            int cropY = relativeY + GEN_CROP_RADIUS;
            int cell = cropY * GEN_CROP_SIDE + cropX;
            int worldX = agentX + relativeX;
            int worldY = agentY + relativeY;
            bool outside = worldX < 0 || worldX >= maze->width || worldY < 0 || worldY >= maze->height;
            if (outside || maze->wall[StateOf(worldX, worldY)]) output[cell] = 1.0f;
            if (!outside && StateOf(worldX, worldY) == maze->goalState) output[plane + cell] = 1.0f;
        }
    }
    output[2 * plane] = (float)(goalX - agentX) / (GEN_MAX_SIZE - 1);
    output[2 * plane + 1] = (float)(goalY - agentY) / (GEN_MAX_SIZE - 1);
}

static void ForwardMlp(
    const GenDqn *dqn,
    const float *parameters,
    GenForward *cache)
{
    int b1 = MlpB1Offset(dqn);
    int w2 = MlpW2Offset(dqn);
    int b2 = MlpB2Offset(dqn);
    for (int h = 0; h < dqn->hiddenSize; h++) {
        float value = parameters[b1 + h];
        int row = MlpW1Offset() + h * dqn->inputSize;
        for (int i = 0; i < dqn->inputSize; i++)
            value += parameters[row + i] * cache->input[i];
        cache->hidden[h] = value > 0.0f ? value : 0.0f;
    }
    for (int action = 0; action < GEN_ACTIONS; action++) {
        float value = parameters[b2 + action];
        int row = w2 + action * dqn->hiddenSize;
        for (int h = 0; h < dqn->hiddenSize; h++)
            value += parameters[row + h] * cache->hidden[h];
        cache->values[action] = value;
    }
}

static void ForwardConv(const float *parameters, GenForward *cache)
{
    int c1w = Conv1WeightsOffset();
    int c1b = Conv1BiasesOffset();
    for (int filter = 0; filter < GEN_CONV1_FILTERS; filter++) {
        for (int y = 0; y < GEN_CONV1_SIDE; y++) {
            for (int x = 0; x < GEN_CONV1_SIDE; x++) {
                float value = parameters[c1b + filter];
                for (int channel = 0; channel < 2; channel++) {
                    for (int ky = 0; ky < GEN_CONV_KERNEL; ky++) {
                        for (int kx = 0; kx < GEN_CONV_KERNEL; kx++) {
                            int inputIndex = channel * GEN_CROP_SIDE * GEN_CROP_SIDE +
                                (y + ky) * GEN_CROP_SIDE + x + kx;
                            int weightIndex = c1w +
                                ((filter * 2 + channel) * GEN_CONV_KERNEL + ky) *
                                GEN_CONV_KERNEL + kx;
                            value += parameters[weightIndex] * cache->input[inputIndex];
                        }
                    }
                }
                int outputIndex = (filter * GEN_CONV1_SIDE + y) * GEN_CONV1_SIDE + x;
                cache->conv1[outputIndex] = value > 0.0f ? value : 0.0f;
            }
        }
    }

    int c2w = Conv2WeightsOffset();
    int c2b = Conv2BiasesOffset();
    for (int filter = 0; filter < GEN_CONV2_FILTERS; filter++) {
        for (int y = 0; y < GEN_CONV2_SIDE; y++) {
            for (int x = 0; x < GEN_CONV2_SIDE; x++) {
                float value = parameters[c2b + filter];
                for (int channel = 0; channel < GEN_CONV1_FILTERS; channel++) {
                    for (int ky = 0; ky < GEN_CONV_KERNEL; ky++) {
                        for (int kx = 0; kx < GEN_CONV_KERNEL; kx++) {
                            int inputY = y * GEN_CONV2_STRIDE + ky;
                            int inputX = x * GEN_CONV2_STRIDE + kx;
                            int inputIndex = (channel * GEN_CONV1_SIDE + inputY) *
                                GEN_CONV1_SIDE + inputX;
                            int weightIndex = c2w +
                                ((filter * GEN_CONV1_FILTERS + channel) * GEN_CONV_KERNEL + ky) *
                                GEN_CONV_KERNEL + kx;
                            value += parameters[weightIndex] * cache->conv1[inputIndex];
                        }
                    }
                }
                int outputIndex = (filter * GEN_CONV2_SIDE + y) * GEN_CONV2_SIDE + x;
                cache->conv2[outputIndex] = value > 0.0f ? value : 0.0f;
            }
        }
    }

    int denseWeights = ConvDenseWeightsOffset();
    int denseBiases = ConvDenseBiasesOffset();
    for (int h = 0; h < GEN_MAX_HIDDEN; h++) {
        float value = parameters[denseBiases + h];
        int row = denseWeights + h * GEN_CONV_DENSE_INPUT;
        for (int i = 0; i < GEN_CONV2_COUNT; i++)
            value += parameters[row + i] * cache->conv2[i];
        value += parameters[row + GEN_CONV2_COUNT] *
            cache->input[2 * GEN_CROP_SIDE * GEN_CROP_SIDE];
        value += parameters[row + GEN_CONV2_COUNT + 1] *
            cache->input[2 * GEN_CROP_SIDE * GEN_CROP_SIDE + 1];
        cache->hidden[h] = value > 0.0f ? value : 0.0f;
    }

    int outputWeights = ConvOutputWeightsOffset();
    int outputBiases = ConvOutputBiasesOffset();
    for (int action = 0; action < GEN_ACTIONS; action++) {
        float value = parameters[outputBiases + action];
        int row = outputWeights + action * GEN_MAX_HIDDEN;
        for (int h = 0; h < GEN_MAX_HIDDEN; h++)
            value += parameters[row + h] * cache->hidden[h];
        cache->values[action] = value;
    }
}

static void ForwardState(
    const GenDqn *dqn,
    const float *parameters,
    const GenMazeView *maze,
    int state,
    GenForward *cache)
{
    Encode(dqn, maze, state, cache->input);
    if (dqn->observation == OBS_CONV) ForwardConv(parameters, cache);
    else ForwardMlp(dqn, parameters, cache);
}

static bool InitializeDqn(
    GenDqn *dqn,
    ObservationKind observation,
    Rng *rng)
{
    memset(dqn, 0, sizeof(*dqn));
    dqn->observation = observation;
    dqn->inputSize = observation == OBS_POSITION ? GEN_POSITION_INPUT : GEN_LAYOUT_INPUT;
    dqn->hiddenSize = observation == OBS_POSITION ? 32 : 64;
    dqn->parameterCount = observation == OBS_CONV ?
        ConvParameterCount() : MlpParameterCount(dqn->inputSize, dqn->hiddenSize);
    size_t bytes = sizeof(float) * (size_t)dqn->parameterCount;
    dqn->online = malloc(bytes);
    dqn->target = malloc(bytes);
    dqn->firstMoment = calloc((size_t)dqn->parameterCount, sizeof(float));
    dqn->secondMoment = calloc((size_t)dqn->parameterCount, sizeof(float));
    dqn->gradient = malloc(bytes);
    if (!dqn->online || !dqn->target || !dqn->firstMoment || !dqn->secondMoment || !dqn->gradient)
        return false;

    for (int i = 0; i < dqn->parameterCount; i++) dqn->online[i] = 0.0f;
    if (observation == OBS_CONV) {
        float c1Scale = sqrtf(2.0f / (2 * GEN_CONV_KERNEL * GEN_CONV_KERNEL));
        float c2Scale = sqrtf(2.0f /
            (GEN_CONV1_FILTERS * GEN_CONV_KERNEL * GEN_CONV_KERNEL));
        float denseScale = sqrtf(2.0f / GEN_CONV_DENSE_INPUT);
        float outputScale = sqrtf(2.0f / GEN_MAX_HIDDEN);
        for (int i = Conv1WeightsOffset(); i < Conv1BiasesOffset(); i++)
            dqn->online[i] = RngNormal(rng) * c1Scale;
        for (int i = Conv2WeightsOffset(); i < Conv2BiasesOffset(); i++)
            dqn->online[i] = RngNormal(rng) * c2Scale;
        for (int i = ConvDenseWeightsOffset(); i < ConvDenseBiasesOffset(); i++)
            dqn->online[i] = RngNormal(rng) * denseScale;
        for (int i = ConvOutputWeightsOffset(); i < ConvOutputBiasesOffset(); i++)
            dqn->online[i] = RngNormal(rng) * outputScale;
    } else {
        int b1 = MlpB1Offset(dqn);
        int w2 = MlpW2Offset(dqn);
        int b2 = MlpB2Offset(dqn);
        float firstScale = sqrtf(2.0f / dqn->inputSize);
        float secondScale = sqrtf(2.0f / dqn->hiddenSize);
        for (int i = 0; i < b1; i++) dqn->online[i] = RngNormal(rng) * firstScale;
        for (int i = w2; i < b2; i++) dqn->online[i] = RngNormal(rng) * secondScale;
    }
    memcpy(dqn->target, dqn->online, bytes);
    return true;
}

static void DestroyDqn(GenDqn *dqn)
{
    free(dqn->online);
    free(dqn->target);
    free(dqn->firstMoment);
    free(dqn->secondMoment);
    free(dqn->gradient);
    memset(dqn, 0, sizeof(*dqn));
}

static Action SelectAction(
    GenDqn *dqn,
    const GenMazeView *maze,
    int state,
    float epsilon,
    Rng *rng)
{
    if (RngFloat(rng) < epsilon) return (Action)RngRange(rng, GEN_ACTIONS);
    GenForward cache;
    ForwardState(dqn, dqn->online, maze, state, &cache);
    Action best[GEN_ACTIONS];
    int count = 1;
    float maximum = cache.values[0];
    best[0] = ACTION_UP;
    for (int action = 1; action < GEN_ACTIONS; action++) {
        if (cache.values[action] > maximum) {
            maximum = cache.values[action];
            count = 1;
            best[0] = (Action)action;
        } else if (cache.values[action] == maximum) {
            best[count++] = (Action)action;
        }
    }
    return best[RngRange(rng, count)];
}

static GenTransition TakeStep(const GenMazeView *maze, int state, Action action)
{
    static const int dx[GEN_ACTIONS] = {0, 1, 0, -1};
    static const int dy[GEN_ACTIONS] = {-1, 0, 1, 0};
    int x = StateX(state);
    int y = StateY(state);
    int nx = x + dx[action];
    int ny = y + dy[action];
    bool blocked = nx < 0 || nx >= maze->width || ny < 0 || ny >= maze->height ||
        maze->wall[StateOf(nx, ny)];
    int nextState = blocked ? state : StateOf(nx, ny);
    bool done = nextState == maze->goalState;
    return (GenTransition){
        *maze,
        state,
        action,
        done ? 100.0f : blocked ? -5.0f : -1.0f,
        nextState,
        done
    };
}

static void ReplayPush(GenDqn *dqn, GenTransition transition)
{
    dqn->replay.entries[dqn->replay.next] = transition;
    dqn->replay.next = (dqn->replay.next + 1) % GEN_REPLAY_CAPACITY;
    if (dqn->replay.count < GEN_REPLAY_CAPACITY) dqn->replay.count++;
}

static float Maximum(const float values[GEN_ACTIONS])
{
    float maximum = values[0];
    for (int action = 1; action < GEN_ACTIONS; action++)
        if (values[action] > maximum) maximum = values[action];
    return maximum;
}

static void AccumulateMlpGradient(
    GenDqn *dqn,
    const GenForward *cache,
    Action action,
    float outputGradient)
{
    int b1 = MlpB1Offset(dqn);
    int w2 = MlpW2Offset(dqn);
    int b2 = MlpB2Offset(dqn);
    dqn->gradient[b2 + action] += outputGradient;
    int outputRow = w2 + action * dqn->hiddenSize;
    for (int h = 0; h < dqn->hiddenSize; h++) {
        dqn->gradient[outputRow + h] += outputGradient * cache->hidden[h];
        if (cache->hidden[h] > 0.0f) {
            float hiddenGradient = outputGradient * dqn->online[outputRow + h];
            dqn->gradient[b1 + h] += hiddenGradient;
            int inputRow = h * dqn->inputSize;
            for (int i = 0; i < dqn->inputSize; i++)
                dqn->gradient[inputRow + i] += hiddenGradient * cache->input[i];
        }
    }
}

static void AccumulateConvGradient(
    GenDqn *dqn,
    const GenForward *cache,
    Action action,
    float outputGradient)
{
    float hiddenGradient[GEN_MAX_HIDDEN] = {0};
    float conv2Gradient[GEN_CONV2_COUNT] = {0};
    float conv1Gradient[GEN_CONV1_COUNT] = {0};
    int outputWeights = ConvOutputWeightsOffset();
    int outputBiases = ConvOutputBiasesOffset();
    int outputRow = outputWeights + action * GEN_MAX_HIDDEN;
    dqn->gradient[outputBiases + action] += outputGradient;
    for (int h = 0; h < GEN_MAX_HIDDEN; h++) {
        dqn->gradient[outputRow + h] += outputGradient * cache->hidden[h];
        if (cache->hidden[h] > 0.0f)
            hiddenGradient[h] = outputGradient * dqn->online[outputRow + h];
    }

    int denseWeights = ConvDenseWeightsOffset();
    int denseBiases = ConvDenseBiasesOffset();
    int displacement = 2 * GEN_CROP_SIDE * GEN_CROP_SIDE;
    for (int h = 0; h < GEN_MAX_HIDDEN; h++) {
        float hiddenGrad = hiddenGradient[h];
        if (hiddenGrad == 0.0f) continue;
        dqn->gradient[denseBiases + h] += hiddenGrad;
        int row = denseWeights + h * GEN_CONV_DENSE_INPUT;
        for (int i = 0; i < GEN_CONV2_COUNT; i++) {
            dqn->gradient[row + i] += hiddenGrad * cache->conv2[i];
            conv2Gradient[i] += hiddenGrad * dqn->online[row + i];
        }
        dqn->gradient[row + GEN_CONV2_COUNT] +=
            hiddenGrad * cache->input[displacement];
        dqn->gradient[row + GEN_CONV2_COUNT + 1] +=
            hiddenGrad * cache->input[displacement + 1];
    }

    int c2w = Conv2WeightsOffset();
    int c2b = Conv2BiasesOffset();
    for (int filter = 0; filter < GEN_CONV2_FILTERS; filter++) {
        for (int y = 0; y < GEN_CONV2_SIDE; y++) {
            for (int x = 0; x < GEN_CONV2_SIDE; x++) {
                int outputIndex = (filter * GEN_CONV2_SIDE + y) * GEN_CONV2_SIDE + x;
                if (cache->conv2[outputIndex] <= 0.0f) continue;
                float gradient = conv2Gradient[outputIndex];
                dqn->gradient[c2b + filter] += gradient;
                for (int channel = 0; channel < GEN_CONV1_FILTERS; channel++) {
                    for (int ky = 0; ky < GEN_CONV_KERNEL; ky++) {
                        for (int kx = 0; kx < GEN_CONV_KERNEL; kx++) {
                            int inputY = y * GEN_CONV2_STRIDE + ky;
                            int inputX = x * GEN_CONV2_STRIDE + kx;
                            int inputIndex = (channel * GEN_CONV1_SIDE + inputY) *
                                GEN_CONV1_SIDE + inputX;
                            int weightIndex = c2w +
                                ((filter * GEN_CONV1_FILTERS + channel) * GEN_CONV_KERNEL + ky) *
                                GEN_CONV_KERNEL + kx;
                            dqn->gradient[weightIndex] += gradient * cache->conv1[inputIndex];
                            conv1Gradient[inputIndex] += gradient * dqn->online[weightIndex];
                        }
                    }
                }
            }
        }
    }

    int c1w = Conv1WeightsOffset();
    int c1b = Conv1BiasesOffset();
    for (int filter = 0; filter < GEN_CONV1_FILTERS; filter++) {
        for (int y = 0; y < GEN_CONV1_SIDE; y++) {
            for (int x = 0; x < GEN_CONV1_SIDE; x++) {
                int outputIndex = (filter * GEN_CONV1_SIDE + y) * GEN_CONV1_SIDE + x;
                if (cache->conv1[outputIndex] <= 0.0f) continue;
                float gradient = conv1Gradient[outputIndex];
                dqn->gradient[c1b + filter] += gradient;
                for (int channel = 0; channel < 2; channel++) {
                    for (int ky = 0; ky < GEN_CONV_KERNEL; ky++) {
                        for (int kx = 0; kx < GEN_CONV_KERNEL; kx++) {
                            int inputIndex = channel * GEN_CROP_SIDE * GEN_CROP_SIDE +
                                (y + ky) * GEN_CROP_SIDE + x + kx;
                            int weightIndex = c1w +
                                ((filter * 2 + channel) * GEN_CONV_KERNEL + ky) *
                                GEN_CONV_KERNEL + kx;
                            dqn->gradient[weightIndex] += gradient * cache->input[inputIndex];
                        }
                    }
                }
            }
        }
    }
}

static void TrainBatch(GenDqn *dqn, Rng *rng)
{
    memset(dqn->gradient, 0, sizeof(float) * (size_t)dqn->parameterCount);
    for (int sample = 0; sample < GEN_BATCH_SIZE; sample++) {
        GenTransition transition = dqn->replay.entries[RngRange(rng, dqn->replay.count)];
        GenForward current;
        ForwardState(
            dqn, dqn->online, &transition.maze, transition.state, &current);

        float target = transition.reward;
        if (!transition.done) {
            GenForward next;
            ForwardState(
                dqn, dqn->target, &transition.maze, transition.nextState, &next);
            target += 0.95f * Maximum(next.values);
        }
        float error = current.values[transition.action] - target;
        float outputGradient = (error > 1.0f ? 1.0f : error < -1.0f ? -1.0f : error) /
            GEN_BATCH_SIZE;
        if (dqn->observation == OBS_CONV)
            AccumulateConvGradient(dqn, &current, transition.action, outputGradient);
        else AccumulateMlpGradient(dqn, &current, transition.action, outputGradient);
    }

    double normSquared = 0.0;
    for (int i = 0; i < dqn->parameterCount; i++)
        normSquared += (double)dqn->gradient[i] * dqn->gradient[i];
    float norm = (float)sqrt(normSquared);
    float gradientScale = norm > 10.0f ? 10.0f / norm : 1.0f;
    dqn->updates++;
    float correction1 = 1.0f - powf(0.9f, (float)dqn->updates);
    float correction2 = 1.0f - powf(0.999f, (float)dqn->updates);
    for (int i = 0; i < dqn->parameterCount; i++) {
        float gradient = dqn->gradient[i] * gradientScale;
        dqn->firstMoment[i] = 0.9f * dqn->firstMoment[i] + 0.1f * gradient;
        dqn->secondMoment[i] = 0.999f * dqn->secondMoment[i] + 0.001f * gradient * gradient;
        float firstHat = dqn->firstMoment[i] / correction1;
        float secondHat = dqn->secondMoment[i] / correction2;
        dqn->online[i] -= 0.001f * firstHat / (sqrtf(secondHat) + 1.0e-8f);
    }
    if (dqn->updates % GEN_TARGET_SYNC == 0)
        memcpy(dqn->target, dqn->online, sizeof(float) * (size_t)dqn->parameterCount);
}

static bool ConvGradientSelfTest(const ExperimentMaze mazes[GEN_MAZE_COUNT])
{
    Rng rng;
    RngSeed(&rng, UINT64_C(0x12345678));
    GenDqn dqn;
    if (!InitializeDqn(&dqn, OBS_CONV, &rng)) {
        DestroyDqn(&dqn);
        return false;
    }
    GenMazeView maze = ViewOfMaze(&mazes[0]);
    GenForward cache;
    ForwardState(&dqn, dqn.online, &maze, mazes[0].startState, &cache);
    const Action action = ACTION_RIGHT;
    const float target = cache.values[action] - 0.25f;
    memset(dqn.gradient, 0, sizeof(float) * (size_t)dqn.parameterCount);
    AccumulateConvGradient(&dqn, &cache, action, cache.values[action] - target);

    int parameter = -1;
    for (int i = Conv1WeightsOffset(); i < Conv1BiasesOffset(); i++) {
        if (fabsf(dqn.gradient[i]) > 1.0e-5f) {
            parameter = i;
            break;
        }
    }
    if (parameter < 0) {
        DestroyDqn(&dqn);
        return false;
    }

    const float epsilon = 1.0e-3f;
    float original = dqn.online[parameter];
    dqn.online[parameter] = original + epsilon;
    GenForward plus;
    ForwardState(&dqn, dqn.online, &maze, mazes[0].startState, &plus);
    float plusError = plus.values[action] - target;
    float plusLoss = 0.5f * plusError * plusError;
    dqn.online[parameter] = original - epsilon;
    GenForward minus;
    ForwardState(&dqn, dqn.online, &maze, mazes[0].startState, &minus);
    float minusError = minus.values[action] - target;
    float minusLoss = 0.5f * minusError * minusError;
    dqn.online[parameter] = original;
    float numerical = (plusLoss - minusLoss) / (2.0f * epsilon);
    float analytic = dqn.gradient[parameter];
    bool passed = fabsf(numerical - analytic) < 2.0e-3f;
    DestroyDqn(&dqn);
    return passed;
}

static GenEpisode RunEpisode(
    GenDqn *dqn,
    const GenMazeView *maze,
    int startState,
    float epsilon,
    bool learn,
    Rng *rng)
{
    GenEpisode result = {0};
    int state = startState;
    for (int step = 0; step < GEN_MAX_STEPS; step++) {
        Action action = SelectAction(dqn, maze, state, epsilon, rng);
        GenTransition transition = TakeStep(maze, state, action);
        if (learn) {
            ReplayPush(dqn, transition);
            dqn->environmentSteps++;
            if (dqn->replay.count >= GEN_REPLAY_WARMUP && dqn->environmentSteps % 4 == 0)
                TrainBatch(dqn, rng);
        }
        result.totalReward += transition.reward;
        result.steps = step + 1;
        state = transition.nextState;
        if (transition.done) { result.reachedGoal = true; break; }
    }
    return result;
}

static GenOptions DefaultOptions(void)
{
    return (GenOptions){5000, 3, 1, "generalization.csv", false, 6, 12, 1, false, 0};
}

static bool ParsePositive(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > 1000000) return false;
    *value = (int)parsed;
    return true;
}

static bool ParseNonNegative(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > 1000000) return false;
    *value = (int)parsed;
    return true;
}

static bool ParseOptions(int argc, char **argv, GenOptions *options)
{
    *options = DefaultOptions();
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->episodes)) return false;
        } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->seeds)) return false;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            int seed;
            if (!ParsePositive(argv[++i], &seed)) return false;
            options->firstSeed = (uint64_t)seed;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            options->csvPath = argv[++i];
        } else if (strcmp(argv[i], "--procedural") == 0) {
            options->procedural = true;
        } else if (strcmp(argv[i], "--min-size") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->proceduralMinSize)) return false;
        } else if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->proceduralMaxSize)) return false;
        } else if (strcmp(argv[i], "--regen-every") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->proceduralRegenEvery)) return false;
        } else if (strcmp(argv[i], "--random-goals") == 0) {
            options->randomGoals = true;
        } else if (strcmp(argv[i], "--min-separation") == 0 && i + 1 < argc) {
            if (!ParseNonNegative(argv[++i], &options->randomGoalsMinSeparation)) return false;
        } else return false;
    }
    if (options->proceduralMinSize < 3 || options->proceduralMaxSize > GEN_MAX_SIZE ||
        options->proceduralMinSize > options->proceduralMaxSize)
        return false;
    if (options->procedural && options->randomGoals) return false;
    if (options->randomGoalsMinSeparation > 0 && !options->randomGoals) return false;
    return true;
}

static GenVideoOptions DefaultVideoOptions(void)
{
    return (GenVideoOptions){
        5000,
        1,
        "video_frames",
        GEN_VIDEO_DEFAULT_FPS,
        GEN_VIDEO_DEFAULT_WIDTH,
        GEN_VIDEO_DEFAULT_HEIGHT,
        GEN_VIDEO_DEFAULT_SECONDS
    };
}

static bool ParseVideoOptions(int argc, char **argv, GenVideoOptions *options)
{
    *options = DefaultVideoOptions();
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->episodes)) return false;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            int seed;
            if (!ParsePositive(argv[++i], &seed)) return false;
            options->seed = (uint64_t)seed;
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            options->framesPath = argv[++i];
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->fps) || options->fps > 120) return false;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->width) || options->width < 640 ||
                options->width > 3840) return false;
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->height) || options->height < 360 ||
                options->height > 2160) return false;
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            if (!ParsePositive(argv[++i], &options->seconds) || options->seconds > 600)
                return false;
        } else return false;
    }
    return options->framesPath[0] != '\0';
}

static int BuildVideoCheckpointEpisodes(
    int episodes,
    int checkpoints[GEN_VIDEO_MAX_CHECKPOINTS])
{
    const int candidates[GEN_VIDEO_MAX_CHECKPOINTS] = {0, 100, 500, 1000, 2500, episodes};
    int count = 0;
    for (int i = 0; i < GEN_VIDEO_MAX_CHECKPOINTS; i++) {
        int candidate = candidates[i];
        if (candidate > episodes) continue;
        if (count == 0 || candidate > checkpoints[count - 1]) checkpoints[count++] = candidate;
    }
    return count;
}

static void FeaturedMazeIndices(int indices[GEN_VIDEO_MAZES])
{
    indices[0] = 0;
    indices[1] = GEN_TRAIN_MAZES;
    indices[2] = GEN_TRAIN_MAZES + GEN_TEST_PER_GROUP;
    indices[3] = GEN_TRAIN_MAZES + 2 * GEN_TEST_PER_GROUP;
}

static GenVideoTrajectory RecordVideoTrajectory(
    GenDqn *dqn,
    const GenMazeView *maze,
    int startState,
    uint64_t evaluationSeed)
{
    GenVideoTrajectory trajectory = {0};
    Rng rng;
    RngSeed(&rng, evaluationSeed);
    int state = startState;
    trajectory.states[0] = state;
    for (int step = 0; step < GEN_MAX_STEPS; step++) {
        Action action = SelectAction(dqn, maze, state, 0.0f, &rng);
        GenTransition transition = TakeStep(maze, state, action);
        trajectory.totalReward += transition.reward;
        trajectory.steps = step + 1;
        state = transition.nextState;
        trajectory.states[trajectory.steps] = state;
        if (transition.done) {
            trajectory.reachedGoal = true;
            break;
        }
    }
    return trajectory;
}

static void DestroyVideoCheckpoints(GenVideoCheckpoint *checkpoints, int count)
{
    for (int i = 0; i < count; i++) {
        free(checkpoints[i].parameters);
        checkpoints[i].parameters = NULL;
    }
}

static bool CaptureVideoCheckpoints(
    const ExperimentMaze mazes[GEN_MAZE_COUNT],
    int episodes,
    uint64_t seed,
    GenVideoCheckpoint checkpoints[GEN_VIDEO_MAX_CHECKPOINTS],
    int *checkpointCount)
{
    int checkpointEpisodes[GEN_VIDEO_MAX_CHECKPOINTS];
    *checkpointCount = BuildVideoCheckpointEpisodes(episodes, checkpointEpisodes);
    memset(checkpoints, 0, sizeof(GenVideoCheckpoint) * GEN_VIDEO_MAX_CHECKPOINTS);
    Rng rng;
    RngSeed(&rng, seed ^ ((uint64_t)OBS_CONV << 48));
    GenDqn dqn;
    if (!InitializeDqn(&dqn, OBS_CONV, &rng)) {
        DestroyDqn(&dqn);
        return false;
    }

    float epsilon = 1.0f;
    int nextCheckpoint = 0;
    size_t parameterBytes = sizeof(float) * (size_t)dqn.parameterCount;
    for (int episode = 0; episode <= episodes; episode++) {
        if (nextCheckpoint < *checkpointCount &&
            episode == checkpointEpisodes[nextCheckpoint]) {
            GenVideoCheckpoint *checkpoint = &checkpoints[nextCheckpoint];
            checkpoint->episode = episode;
            checkpoint->parameters = malloc(parameterBytes);
            if (!checkpoint->parameters) {
                DestroyVideoCheckpoints(checkpoints, *checkpointCount);
                DestroyDqn(&dqn);
                return false;
            }
            memcpy(checkpoint->parameters, dqn.online, parameterBytes);
            nextCheckpoint++;
        }
        if (episode == episodes) break;
        int mazeIndex = RngRange(&rng, GEN_TRAIN_MAZES);
        GenMazeView trainMaze = ViewOfMaze(&mazes[mazeIndex]);
        RunEpisode(&dqn, &trainMaze, mazes[mazeIndex].startState, epsilon, true, &rng);
        epsilon *= 0.9995f;
        if (epsilon < 0.05f) epsilon = 0.05f;
    }

    int mazeIndices[GEN_VIDEO_MAZES];
    FeaturedMazeIndices(mazeIndices);
    for (int checkpointIndex = 0; checkpointIndex < *checkpointCount; checkpointIndex++) {
        memcpy(dqn.online, checkpoints[checkpointIndex].parameters, parameterBytes);
        for (int mazeSlot = 0; mazeSlot < GEN_VIDEO_MAZES; mazeSlot++) {
            int mazeIndex = mazeIndices[mazeSlot];
            uint64_t evaluationSeed = seed ^ (uint64_t)(mazeIndex + 1) ^
                UINT64_C(0x6a09e667);
            GenMazeView evalMaze = ViewOfMaze(&mazes[mazeIndex]);
            checkpoints[checkpointIndex].trajectories[mazeSlot] = RecordVideoTrajectory(
                &dqn, &evalMaze, mazes[mazeIndex].startState, evaluationSeed);
        }
    }
    DestroyDqn(&dqn);
    return nextCheckpoint == *checkpointCount;
}

static const char *ObservationName(ObservationKind kind)
{
    if (kind == OBS_CONV) return "conv_layout";
    return kind == OBS_LAYOUT ? "layout_aware" : "position_only";
}

static int RunOne(
    FILE *csv,
    const ExperimentMaze mazes[GEN_MAZE_COUNT],
    ObservationKind observation,
    const GenOptions *options,
    uint64_t seed)
{
    Rng rng;
    RngSeed(&rng, seed ^ ((uint64_t)observation << 48));
    GenDqn dqn;
    if (!InitializeDqn(&dqn, observation, &rng)) {
        DestroyDqn(&dqn);
        return 1;
    }
    char trainMode[32];
    if (options->procedural) {
        snprintf(trainMode, sizeof(trainMode), "procedural_%dto%d",
            options->proceduralMinSize, options->proceduralMaxSize);
    } else if (options->randomGoals) {
        if (options->randomGoalsMinSeparation > 0) {
            snprintf(trainMode, sizeof(trainMode), "fixed_%d_random_goals_sep%d",
                GEN_TRAIN_MAZES, options->randomGoalsMinSeparation);
        } else {
            snprintf(trainMode, sizeof(trainMode), "fixed_%d_random_goals", GEN_TRAIN_MAZES);
        }
    } else {
        snprintf(trainMode, sizeof(trainMode), "fixed_%d", GEN_TRAIN_MAZES);
    }

    float epsilon = 1.0f;
    GenMazeView proceduralMaze = {0};
    int proceduralStart = 0;
    /* Records every distinct procedural maze this run actually trained on
       (up to GEN_POOL_CAP), so we can separately measure whether the network
       fits its own training pool versus generalizes beyond it. */
    GenMazeView poolMazes[GEN_POOL_CAP];
    int poolStarts[GEN_POOL_CAP];
    int poolCount = 0;
    clock_t start = clock();
    for (int episode = 0; episode < options->episodes; episode++) {
        if (options->procedural) {
            if (episode % options->proceduralRegenEvery == 0) {
                GenerateProceduralMaze(&proceduralMaze, &proceduralStart,
                    options->proceduralMinSize, options->proceduralMaxSize, &rng);
                if (poolCount < GEN_POOL_CAP) {
                    poolMazes[poolCount] = proceduralMaze;
                    poolStarts[poolCount] = proceduralStart;
                    poolCount++;
                }
            }
            RunEpisode(&dqn, &proceduralMaze, proceduralStart, epsilon, true, &rng);
        } else if (options->randomGoals) {
            int mazeIndex = RngRange(&rng, GEN_TRAIN_MAZES);
            GenMazeView trainMaze = ViewOfMaze(&mazes[mazeIndex]);
            int randomStart;
            int randomGoal;
            RandomizeStartGoal(&mazes[mazeIndex], &randomStart, &randomGoal,
                options->randomGoalsMinSeparation, &rng);
            trainMaze.goalState = randomGoal;
            if (poolCount < GEN_POOL_CAP) {
                poolMazes[poolCount] = trainMaze;
                poolStarts[poolCount] = randomStart;
                poolCount++;
            }
            RunEpisode(&dqn, &trainMaze, randomStart, epsilon, true, &rng);
        } else {
            int mazeIndex = RngRange(&rng, GEN_TRAIN_MAZES);
            GenMazeView trainMaze = ViewOfMaze(&mazes[mazeIndex]);
            RunEpisode(&dqn, &trainMaze, mazes[mazeIndex].startState, epsilon, true, &rng);
        }
        epsilon *= 0.9995f;
        if (epsilon < 0.05f) epsilon = 0.05f;
    }
    double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;

    int groupSuccess[4] = {0};
    int groupTotal[4] = {0};
    for (int mazeIndex = 0; mazeIndex < GEN_MAZE_COUNT; mazeIndex++) {
        Rng evaluationRng;
        RngSeed(&evaluationRng, seed ^ (uint64_t)(mazeIndex + 1) ^ UINT64_C(0x6a09e667));
        GenMazeView evalMaze = ViewOfMaze(&mazes[mazeIndex]);
        GenEpisode evaluation = RunEpisode(
            &dqn, &evalMaze, mazes[mazeIndex].startState, 0.0f, false, &evaluationRng);
        const ExperimentMaze *maze = &mazes[mazeIndex];
        int gap = evaluation.reachedGoal ? evaluation.steps - maze->optimalSteps : -1;
        fprintf(csv, "%llu,%s,%s,%s,%s,%llu,%dx%d,%d,%d,%d,%d,%.1f,%.3f,%d\n",
            (unsigned long long)seed,
            ObservationName(observation),
            trainMode,
            maze->split,
            maze->name,
            (unsigned long long)maze->generationSeed,
            maze->width,
            maze->height,
            maze->optimalSteps,
            evaluation.reachedGoal ? 1 : 0,
            evaluation.steps,
            gap,
            evaluation.totalReward,
            elapsed,
            dqn.parameterCount);
        int group = mazeIndex < GEN_TRAIN_MAZES ? 0 :
            mazeIndex < GEN_TRAIN_MAZES + GEN_TEST_PER_GROUP ? 1 :
            mazeIndex < GEN_TRAIN_MAZES + 2 * GEN_TEST_PER_GROUP ? 2 : 3;
        groupTotal[group]++;
        if (evaluation.reachedGoal) groupSuccess[group]++;
    }
    int poolSuccess = 0;
    if (options->procedural || options->randomGoals) {
        for (int p = 0; p < poolCount; p++) {
            Rng poolRng;
            RngSeed(&poolRng, seed ^ (uint64_t)p ^ UINT64_C(0x9e3779b97f4a7c15));
            GenEpisode poolEval = RunEpisode(
                &dqn, &poolMazes[p], poolStarts[p], 0.0f, false, &poolRng);
            if (poolEval.reachedGoal) poolSuccess++;
        }
    }

    printf("seed=%llu observation=%s train_mode=%s train=%d/%d same=%d/%d smaller=%d/%d larger=%d/%d",
        (unsigned long long)seed,
        ObservationName(observation),
        trainMode,
        groupSuccess[0], groupTotal[0], groupSuccess[1], groupTotal[1],
        groupSuccess[2], groupTotal[2], groupSuccess[3], groupTotal[3]);
    if (options->procedural || options->randomGoals)
        printf(" pool_fit=%d/%d", poolSuccess, poolCount);
    printf(" time=%.0fms\n", elapsed);
    DestroyDqn(&dqn);
    return 0;
}

int RunGeneralizationExperiment(int argc, char **argv)
{
    GenOptions options;
    if (!ParseOptions(argc, argv, &options)) {
        fprintf(stderr, "Usage: %s --generalization [--episodes N] [--seeds N] [--seed N] [--csv FILE] "
            "[--procedural] [--min-size N] [--max-size N] [--regen-every N] "
            "[--random-goals [--min-separation N]]\n", argv[0]);
        return 2;
    }
    ExperimentMaze mazes[GEN_MAZE_COUNT];
    BuildMazeSuite(mazes, UINT64_C(20260830));
    for (int i = 0; i < GEN_MAZE_COUNT; i++) {
        if (mazes[i].optimalSteps < 1) {
            fprintf(stderr, "Could not generate solvable maze %s\n", mazes[i].name);
            return 1;
        }
    }
    FILE *csv = fopen(options.csvPath, "w");
    if (!csv) {
        fprintf(stderr, "Could not open CSV output: %s\n", options.csvPath);
        return 1;
    }
    fprintf(csv, "seed,observation,train_mode,split,maze,maze_seed,size,optimal_steps,success,steps,excess_steps,return,training_ms,parameters\n");
    int status = 0;
    for (int offset = 0; offset < options.seeds && status == 0; offset++) {
        uint64_t seed = options.firstSeed + (uint64_t)offset;
        status = RunOne(csv, mazes, OBS_POSITION, &options, seed);
        if (status == 0) status = RunOne(csv, mazes, OBS_LAYOUT, &options, seed);
        if (status == 0) status = RunOne(csv, mazes, OBS_CONV, &options, seed);
        fflush(csv);
    }
    fclose(csv);
    if (status == 0) printf("Wrote generalization results to %s\n", options.csvPath);
    return status;
}

static int ScaledFontSize(int base, int height)
{
    int size = base * height / GEN_VIDEO_DEFAULT_HEIGHT;
    return size < 10 ? 10 : size;
}

static void DrawCenteredText(const char *text, int y, int fontSize, Color color, int width)
{
    DrawText(text, (width - MeasureText(text, fontSize)) / 2, y, fontSize, color);
}

static void DrawVideoTitle(int width, int height, uint64_t seed)
{
    Color background = {13, 20, 33, 255};
    Color accent = {76, 201, 240, 255};
    ClearBackground(background);
    int titleSize = ScaledFontSize(48, height);
    int subtitleSize = ScaledFontSize(24, height);
    DrawCenteredText("CAN A DQN LEARN TO NAVIGATE NEW MAZES?",
        height * 28 / 100, titleSize, RAYWHITE, width);
    DrawCenteredText("Convolutional policy at six training checkpoints",
        height * 43 / 100, subtitleSize, accent, width);
    char subtitle[128];
    snprintf(subtitle, sizeof(subtitle),
        "Frozen greedy evaluation | Seed %llu | 16 training mazes",
        (unsigned long long)seed);
    DrawCenteredText(subtitle, height * 51 / 100,
        ScaledFontSize(19, height), LIGHTGRAY, width);
    DrawRectangle(width * 30 / 100, height * 62 / 100, width * 40 / 100,
        height / 100, accent);
}

static void DrawMazeTrajectory(
    const ExperimentMaze *maze,
    const GenVideoTrajectory *trajectory,
    int visibleStep,
    int width,
    int height)
{
    float maxMazeWidth = width * 0.43f;
    float maxMazeHeight = height * 0.70f;
    float cell = fminf(maxMazeWidth / maze->width, maxMazeHeight / maze->height);
    float mazeWidth = cell * maze->width;
    float mazeHeight = cell * maze->height;
    float originX = width * 0.285f - mazeWidth * 0.5f;
    float originY = height * 0.54f - mazeHeight * 0.5f;
    Color wallColor = {36, 50, 69, 255};
    Color floorColor = {237, 242, 247, 255};
    Color gridColor = {188, 198, 211, 255};
    for (int y = 0; y < maze->height; y++) {
        for (int x = 0; x < maze->width; x++) {
            int state = StateOf(x, y);
            Color cellColor = maze->wall[state] ? wallColor : floorColor;
            if (state == maze->startState) cellColor = (Color){61, 133, 224, 255};
            if (state == maze->goalState) cellColor = (Color){56, 176, 112, 255};
            Rectangle rectangle = {originX + x * cell, originY + y * cell, cell, cell};
            DrawRectangleRec(rectangle, cellColor);
            DrawRectangleLinesEx(rectangle, 1.0f, gridColor);
        }
    }

    Color trailColor = {255, 159, 67, 185};
    for (int i = 1; i <= visibleStep; i++) {
        int previous = trajectory->states[i - 1];
        int current = trajectory->states[i];
        Vector2 from = {
            originX + (StateX(previous) + 0.5f) * cell,
            originY + (StateY(previous) + 0.5f) * cell
        };
        Vector2 to = {
            originX + (StateX(current) + 0.5f) * cell,
            originY + (StateY(current) + 0.5f) * cell
        };
        DrawLineEx(from, to, fmaxf(2.0f, cell * 0.12f), trailColor);
    }
    int state = trajectory->states[visibleStep];
    Vector2 agent = {
        originX + (StateX(state) + 0.5f) * cell,
        originY + (StateY(state) + 0.5f) * cell
    };
    DrawCircleV(agent, cell * 0.28f, (Color){231, 76, 60, 255});
    DrawCircleLines((int)agent.x, (int)agent.y, cell * 0.28f, MAROON);
}

static const char *ReadableSplit(const char *split)
{
    if (strcmp(split, "train") == 0) return "TRAINING MAZE";
    if (strcmp(split, "heldout_same") == 0) return "UNSEEN 10 x 10";
    if (strcmp(split, "heldout_smaller") == 0) return "UNSEEN 8 x 8";
    return "UNSEEN 12 x 12";
}

static void DrawVideoScene(
    const ExperimentMaze *maze,
    const GenVideoCheckpoint *checkpoint,
    const GenVideoTrajectory *trajectory,
    int visibleStep,
    int totalEpisodes,
    int width,
    int height)
{
    Color background = {247, 249, 252, 255};
    Color navy = {23, 37, 61, 255};
    Color accent = {28, 126, 214, 255};
    Color success = {31, 145, 90, 255};
    Color failure = {196, 65, 55, 255};
    ClearBackground(background);
    DrawRectangle(0, 0, width, height * 11 / 100, navy);
    char text[160];
    snprintf(text, sizeof(text), "CONVOLUTIONAL DQN  |  EPISODE %d / %d",
        checkpoint->episode, totalEpisodes);
    DrawText(text, width * 4 / 100, height * 35 / 1000,
        ScaledFontSize(26, height), RAYWHITE);
    DrawText("FROZEN GREEDY POLICY", width * 72 / 100, height * 4 / 100,
        ScaledFontSize(18, height), (Color){151, 210, 255, 255});

    DrawMazeTrajectory(maze, trajectory, visibleStep, width, height);
    int panelX = width * 55 / 100;
    int panelY = height * 18 / 100;
    int heading = ScaledFontSize(26, height);
    int body = ScaledFontSize(20, height);
    DrawText(ReadableSplit(maze->split), panelX, panelY, heading, accent);
    snprintf(text, sizeof(text), "%s  |  %d x %d", maze->name, maze->width, maze->height);
    DrawText(text, panelX, panelY + height * 7 / 100, body, navy);
    snprintf(text, sizeof(text), "BFS optimum: %d steps", maze->optimalSteps);
    DrawText(text, panelX, panelY + height * 15 / 100, body, DARKGRAY);
    snprintf(text, sizeof(text), "Animated step: %d / %d", visibleStep, trajectory->steps);
    DrawText(text, panelX, panelY + height * 21 / 100, body, DARKGRAY);
    snprintf(text, sizeof(text), "Episode return: %.0f", trajectory->totalReward);
    DrawText(text, panelX, panelY + height * 27 / 100, body, DARKGRAY);
    Color outcomeColor = trajectory->reachedGoal ? success : failure;
    snprintf(text, sizeof(text), "OUTCOME: %s",
        trajectory->reachedGoal ? "SUCCESS" : "TIMEOUT AT 200 STEPS");
    DrawText(text, panelX, panelY + height * 38 / 100, heading, outcomeColor);
    if (trajectory->reachedGoal) {
        snprintf(text, sizeof(text), "Optimality gap: %+d steps",
            trajectory->steps - maze->optimalSteps);
        DrawText(text, panelX, panelY + height * 45 / 100, body, outcomeColor);
    } else {
        DrawText("Long or cyclic paths are compressed in time.", panelX,
            panelY + height * 45 / 100, ScaledFontSize(17, height), GRAY);
    }

    float progress = totalEpisodes > 0 ? (float)checkpoint->episode / totalEpisodes : 0.0f;
    int barX = width * 55 / 100;
    int barY = height * 79 / 100;
    int barWidth = width * 38 / 100;
    DrawRectangle(barX, barY, barWidth, height * 2 / 100, (Color){211, 219, 230, 255});
    DrawRectangle(barX, barY, (int)(barWidth * progress), height * 2 / 100, accent);
    DrawText("TRAINING PROGRESS", barX, barY - height * 5 / 100,
        ScaledFontSize(16, height), GRAY);
    DrawText("Same four maze indices at every checkpoint", panelX, height * 90 / 100,
        ScaledFontSize(17, height), GRAY);
}

static void DrawVideoSummary(
    const ExperimentMaze mazes[GEN_MAZE_COUNT],
    const GenVideoCheckpoint *finalCheckpoint,
    int width,
    int height)
{
    Color background = {13, 20, 33, 255};
    Color accent = {76, 201, 240, 255};
    ClearBackground(background);
    DrawCenteredText("WHAT DID THE POLICY LEARN?", height * 12 / 100,
        ScaledFontSize(40, height), RAYWHITE, width);
    DrawCenteredText("Final checkpoint on the fixed, unbiased video set",
        height * 22 / 100, ScaledFontSize(20, height), accent, width);
    int mazeIndices[GEN_VIDEO_MAZES];
    FeaturedMazeIndices(mazeIndices);
    for (int slot = 0; slot < GEN_VIDEO_MAZES; slot++) {
        const ExperimentMaze *maze = &mazes[mazeIndices[slot]];
        const GenVideoTrajectory *trajectory = &finalCheckpoint->trajectories[slot];
        int y = height * (35 + slot * 11) / 100;
        char text[180];
        snprintf(text, sizeof(text), "%-16s  %-14s  %s",
            ReadableSplit(maze->split), maze->name,
            trajectory->reachedGoal ? "SUCCESS" : "TIMEOUT");
        DrawText(text, width * 18 / 100, y, ScaledFontSize(22, height),
            trajectory->reachedGoal ? (Color){74, 222, 128, 255} : (Color){248, 113, 113, 255});
    }
    DrawCenteredText("Spatial sharing helps, but reliable held-out planning remains unsolved.",
        height * 84 / 100, ScaledFontSize(20, height), LIGHTGRAY, width);
}

static bool ExportVideoFrame(RenderTexture2D target, const char *framesPath, int frame)
{
    char fileName[1024];
    int length = snprintf(fileName, sizeof(fileName), "%s/frame_%05d.png", framesPath, frame);
    if (length < 0 || (size_t)length >= sizeof(fileName)) return false;
    Image image = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&image);
    bool exported = ExportImage(image, fileName);
    UnloadImage(image);
    return exported;
}

static bool WriteVideoManifest(
    const GenVideoOptions *options,
    int frames,
    int checkpoints)
{
    char fileName[1024];
    int length = snprintf(fileName, sizeof(fileName), "%s/manifest.txt", options->framesPath);
    if (length < 0 || (size_t)length >= sizeof(fileName)) return false;
    FILE *manifest = fopen(fileName, "w");
    if (!manifest) return false;
    fprintf(manifest, "fps=%d\nwidth=%d\nheight=%d\nframes=%d\ncheckpoints=%d\n",
        options->fps, options->width, options->height, frames, checkpoints);
    return fclose(manifest) == 0;
}

static bool RenderVideoFrames(
    const GenVideoOptions *options,
    const ExperimentMaze mazes[GEN_MAZE_COUNT],
    const GenVideoCheckpoint checkpoints[GEN_VIDEO_MAX_CHECKPOINTS],
    int checkpointCount)
{
    if (!DirectoryExists(options->framesPath)) {
        fprintf(stderr, "Frame directory does not exist: %s\n", options->framesPath);
        return false;
    }
    int totalFrames = options->fps * options->seconds;
    int titleFrames = options->fps * 2;
    int summaryFrames = options->fps * 3;
    if (titleFrames + summaryFrames >= totalFrames) {
        titleFrames = options->fps;
        summaryFrames = options->fps;
    }
    int contentFrames = totalFrames - titleFrames - summaryFrames;
    int sceneCount = checkpointCount * GEN_VIDEO_MAZES;
    int mazeIndices[GEN_VIDEO_MAZES];
    FeaturedMazeIndices(mazeIndices);

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(1, 1, "Maze RL video renderer");
    RenderTexture2D target = LoadRenderTexture(options->width, options->height);
    if (!IsRenderTextureValid(target)) {
        CloseWindow();
        return false;
    }
    bool success = true;
    for (int frame = 0; frame < totalFrames && success; frame++) {
        BeginTextureMode(target);
        if (frame < titleFrames) {
            DrawVideoTitle(options->width, options->height, options->seed);
        } else if (frame >= titleFrames + contentFrames) {
            DrawVideoSummary(mazes, &checkpoints[checkpointCount - 1],
                options->width, options->height);
        } else {
            int contentFrame = frame - titleFrames;
            int scene = contentFrame * sceneCount / contentFrames;
            if (scene >= sceneCount) scene = sceneCount - 1;
            int sceneStart = scene * contentFrames / sceneCount;
            int sceneEnd = (scene + 1) * contentFrames / sceneCount;
            int sceneFrame = contentFrame - sceneStart;
            int sceneFrames = sceneEnd - sceneStart;
            int checkpointIndex = scene / GEN_VIDEO_MAZES;
            int mazeSlot = scene % GEN_VIDEO_MAZES;
            const GenVideoTrajectory *trajectory =
                &checkpoints[checkpointIndex].trajectories[mazeSlot];
            int visibleStep = sceneFrames > 1 ?
                trajectory->steps * sceneFrame / (sceneFrames - 1) : trajectory->steps;
            if (visibleStep > trajectory->steps) visibleStep = trajectory->steps;
            DrawVideoScene(
                &mazes[mazeIndices[mazeSlot]],
                &checkpoints[checkpointIndex],
                trajectory,
                visibleStep,
                options->episodes,
                options->width,
                options->height);
        }
        EndTextureMode();
        success = ExportVideoFrame(target, options->framesPath, frame);
        if (frame % (options->fps * 5) == 0 || frame + 1 == totalFrames)
            printf("Rendered frame %d/%d\n", frame + 1, totalFrames);
    }
    UnloadRenderTexture(target);
    CloseWindow();
    return success && WriteVideoManifest(options, totalFrames, checkpointCount);
}

int RunGeneralizationVideo(int argc, char **argv)
{
    GenVideoOptions options;
    if (!ParseVideoOptions(argc, argv, &options)) {
        fprintf(stderr, "Usage: %s --generalization-video [--episodes N] [--seed N] "
            "[--frames DIR] [--fps N] [--width N] [--height N] [--seconds N]\n",
            argv[0]);
        return 2;
    }
    ExperimentMaze mazes[GEN_MAZE_COUNT];
    BuildMazeSuite(mazes, UINT64_C(20260830));
    GenVideoCheckpoint checkpoints[GEN_VIDEO_MAX_CHECKPOINTS];
    int checkpointCount = 0;
    printf("Training convolutional DQN and capturing checkpoints...\n");
    if (!CaptureVideoCheckpoints(
            mazes, options.episodes, options.seed, checkpoints, &checkpointCount)) {
        fprintf(stderr, "Could not capture video checkpoints.\n");
        return 1;
    }
    printf("Captured %d checkpoints. Rendering %d seconds at %d FPS...\n",
        checkpointCount, options.seconds, options.fps);
    bool rendered = RenderVideoFrames(&options, mazes, checkpoints, checkpointCount);
    DestroyVideoCheckpoints(checkpoints, checkpointCount);
    if (!rendered) {
        fprintf(stderr, "Could not render video frames.\n");
        return 1;
    }
    printf("Wrote video frames to %s\n", options.framesPath);
    return 0;
}

bool GeneralizationRunSelfTests(void)
{
    ExperimentMaze mazes[GEN_MAZE_COUNT];
    BuildMazeSuite(mazes, UINT64_C(20260830));
    for (int i = 0; i < GEN_MAZE_COUNT; i++) {
        if (mazes[i].width < 3 || mazes[i].width > GEN_MAX_SIZE ||
            mazes[i].height < 3 || mazes[i].height > GEN_MAX_SIZE ||
            mazes[i].optimalSteps != ShortestPath(&mazes[i]) ||
            mazes[i].wall[mazes[i].startState] || mazes[i].wall[mazes[i].goalState])
            return false;
    }
    GenMazeView firstMaze = ViewOfMaze(&mazes[0]);
    GenTransition wall = TakeStep(&firstMaze, mazes[0].startState, ACTION_UP);
    if (wall.nextState != mazes[0].startState || wall.reward != -5.0f || wall.done)
        return false;
    if (GEN_LAYOUT_INPUT != 340 || MlpParameterCount(GEN_LAYOUT_INPUT, 64) != 22084 ||
        GEN_CONV1_SIDE != 11 || GEN_CONV2_SIDE != 5 || ConvParameterCount() != 13624)
        return false;
    if (!ConvGradientSelfTest(mazes)) return false;

    Rng proceduralRng;
    RngSeed(&proceduralRng, UINT64_C(0x9e3779b97f4a7c15));
    for (int i = 0; i < 200; i++) {
        GenMazeView view;
        int startState;
        GenerateProceduralMaze(&view, &startState, 6, 12, &proceduralRng);
        if (view.width < 6 || view.width > 12 || view.height != view.width) return false;
        if (view.wall[startState] || view.wall[view.goalState]) return false;
        if (startState == view.goalState) return false;
        if (ShortestPathGeneric(view.width, view.height, view.wall, startState, view.goalState) < 0)
            return false;
    }
    GenMazeView fixedSizeView;
    int fixedSizeStart;
    GenerateProceduralMaze(&fixedSizeView, &fixedSizeStart, 8, 8, &proceduralRng);
    if (fixedSizeView.width != 8 || fixedSizeView.height != 8) return false;

    int checkpointEpisodes[GEN_VIDEO_MAX_CHECKPOINTS];
    int checkpointCount = BuildVideoCheckpointEpisodes(5000, checkpointEpisodes);
    const int expected[GEN_VIDEO_MAX_CHECKPOINTS] = {0, 100, 500, 1000, 2500, 5000};
    if (checkpointCount != GEN_VIDEO_MAX_CHECKPOINTS) return false;
    for (int i = 0; i < checkpointCount; i++)
        if (checkpointEpisodes[i] != expected[i]) return false;
    checkpointCount = BuildVideoCheckpointEpisodes(20, checkpointEpisodes);
    if (checkpointCount != 2 || checkpointEpisodes[0] != 0 || checkpointEpisodes[1] != 20)
        return false;

    int mazeIndices[GEN_VIDEO_MAZES];
    FeaturedMazeIndices(mazeIndices);
    if (mazeIndices[0] != 0 || mazeIndices[1] != GEN_TRAIN_MAZES ||
        mazeIndices[2] != GEN_TRAIN_MAZES + GEN_TEST_PER_GROUP ||
        mazeIndices[3] != GEN_TRAIN_MAZES + 2 * GEN_TEST_PER_GROUP)
        return false;
    if (strcmp(mazes[mazeIndices[0]].split, "train") != 0 ||
        strcmp(mazes[mazeIndices[1]].split, "heldout_same") != 0 ||
        strcmp(mazes[mazeIndices[2]].split, "heldout_smaller") != 0 ||
        strcmp(mazes[mazeIndices[3]].split, "heldout_larger") != 0)
        return false;

    Rng videoRng;
    RngSeed(&videoRng, UINT64_C(1) ^ ((uint64_t)OBS_CONV << 48));
    GenDqn videoDqn;
    if (!InitializeDqn(&videoDqn, OBS_CONV, &videoRng)) {
        DestroyDqn(&videoDqn);
        return false;
    }
    int updatesBefore = videoDqn.updates;
    int stepsBefore = videoDqn.environmentSteps;
    int replayBefore = videoDqn.replay.count;
    float parameterBefore = videoDqn.online[0];
    uint64_t evaluationSeed = UINT64_C(1) ^ (uint64_t)(mazeIndices[0] + 1) ^
        UINT64_C(0x6a09e667);
    GenMazeView firstMazeView = ViewOfMaze(&mazes[mazeIndices[0]]);
    GenVideoTrajectory first = RecordVideoTrajectory(
        &videoDqn, &firstMazeView, mazes[mazeIndices[0]].startState, evaluationSeed);
    GenVideoTrajectory second = RecordVideoTrajectory(
        &videoDqn, &firstMazeView, mazes[mazeIndices[0]].startState, evaluationSeed);
    bool trajectoryValid = first.steps >= 1 && first.steps <= GEN_MAX_STEPS &&
        first.steps == second.steps && first.reachedGoal == second.reachedGoal &&
        first.totalReward == second.totalReward;
    for (int i = 0; trajectoryValid && i <= first.steps; i++) {
        if (first.states[i] != second.states[i] || first.states[i] < 0 ||
            first.states[i] >= GEN_MAX_CELLS)
            trajectoryValid = false;
    }
    bool frozen = videoDqn.updates == updatesBefore &&
        videoDqn.environmentSteps == stepsBefore && videoDqn.replay.count == replayBefore &&
        videoDqn.online[0] == parameterBefore;
    DestroyDqn(&videoDqn);
    if (!trajectoryValid || !frozen) return false;
    return true;
}
