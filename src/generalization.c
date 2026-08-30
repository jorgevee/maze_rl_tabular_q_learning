#include "generalization.h"
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

typedef enum {
    OBS_POSITION,
    OBS_LAYOUT
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

typedef struct {
    int mazeIndex;
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
    const ExperimentMaze *mazes;
} GenDqn;

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
} GenOptions;

static int StateOf(int x, int y) { return y * GEN_MAX_SIZE + x; }
static int StateX(int state) { return state % GEN_MAX_SIZE; }
static int StateY(int state) { return state / GEN_MAX_SIZE; }

static int ShortestPath(const ExperimentMaze *maze)
{
    int distance[GEN_MAX_CELLS];
    int queue[GEN_MAX_CELLS];
    for (int i = 0; i < GEN_MAX_CELLS; i++) distance[i] = -1;
    int front = 0;
    int back = 0;
    queue[back++] = maze->startState;
    distance[maze->startState] = 0;
    static const int dx[GEN_ACTIONS] = {0, 1, 0, -1};
    static const int dy[GEN_ACTIONS] = {-1, 0, 1, 0};
    while (front < back) {
        int state = queue[front++];
        if (state == maze->goalState) return distance[state];
        int x = StateX(state);
        int y = StateY(state);
        for (int action = 0; action < GEN_ACTIONS; action++) {
            int nx = x + dx[action];
            int ny = y + dy[action];
            if (nx < 0 || nx >= maze->width || ny < 0 || ny >= maze->height) continue;
            int next = StateOf(nx, ny);
            if (maze->wall[next] || distance[next] >= 0) continue;
            distance[next] = distance[state] + 1;
            queue[back++] = next;
        }
    }
    return -1;
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

static int ParameterCount(int inputSize, int hiddenSize)
{
    return hiddenSize * inputSize + hiddenSize + GEN_ACTIONS * hiddenSize + GEN_ACTIONS;
}

static int W1Offset(void) { return 0; }
static int B1Offset(const GenDqn *dqn) { return dqn->hiddenSize * dqn->inputSize; }
static int W2Offset(const GenDqn *dqn) { return B1Offset(dqn) + dqn->hiddenSize; }
static int B2Offset(const GenDqn *dqn) { return W2Offset(dqn) + GEN_ACTIONS * dqn->hiddenSize; }

static void Encode(
    const GenDqn *dqn,
    int mazeIndex,
    int state,
    float output[GEN_MAX_INPUT])
{
    memset(output, 0, sizeof(float) * (size_t)dqn->inputSize);
    if (dqn->observation == OBS_POSITION) {
        output[state] = 1.0f;
        return;
    }

    const ExperimentMaze *maze = &dqn->mazes[mazeIndex];
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

static void Forward(
    const GenDqn *dqn,
    const float *parameters,
    const float input[GEN_MAX_INPUT],
    float hidden[GEN_MAX_HIDDEN],
    float values[GEN_ACTIONS])
{
    int b1 = B1Offset(dqn);
    int w2 = W2Offset(dqn);
    int b2 = B2Offset(dqn);
    for (int h = 0; h < dqn->hiddenSize; h++) {
        float value = parameters[b1 + h];
        int row = W1Offset() + h * dqn->inputSize;
        for (int i = 0; i < dqn->inputSize; i++) value += parameters[row + i] * input[i];
        hidden[h] = value > 0.0f ? value : 0.0f;
    }
    for (int action = 0; action < GEN_ACTIONS; action++) {
        float value = parameters[b2 + action];
        int row = w2 + action * dqn->hiddenSize;
        for (int h = 0; h < dqn->hiddenSize; h++) value += parameters[row + h] * hidden[h];
        values[action] = value;
    }
}

static bool InitializeDqn(
    GenDqn *dqn,
    ObservationKind observation,
    const ExperimentMaze *mazes,
    Rng *rng)
{
    memset(dqn, 0, sizeof(*dqn));
    dqn->observation = observation;
    dqn->mazes = mazes;
    dqn->inputSize = observation == OBS_LAYOUT ? GEN_LAYOUT_INPUT : GEN_POSITION_INPUT;
    dqn->hiddenSize = observation == OBS_LAYOUT ? 64 : 32;
    dqn->parameterCount = ParameterCount(dqn->inputSize, dqn->hiddenSize);
    size_t bytes = sizeof(float) * (size_t)dqn->parameterCount;
    dqn->online = malloc(bytes);
    dqn->target = malloc(bytes);
    dqn->firstMoment = calloc((size_t)dqn->parameterCount, sizeof(float));
    dqn->secondMoment = calloc((size_t)dqn->parameterCount, sizeof(float));
    dqn->gradient = malloc(bytes);
    if (!dqn->online || !dqn->target || !dqn->firstMoment || !dqn->secondMoment || !dqn->gradient)
        return false;

    int b1 = B1Offset(dqn);
    int w2 = W2Offset(dqn);
    int b2 = B2Offset(dqn);
    float firstScale = sqrtf(2.0f / dqn->inputSize);
    float secondScale = sqrtf(2.0f / dqn->hiddenSize);
    for (int i = 0; i < dqn->parameterCount; i++) dqn->online[i] = 0.0f;
    for (int i = 0; i < b1; i++) dqn->online[i] = RngNormal(rng) * firstScale;
    for (int i = w2; i < b2; i++) dqn->online[i] = RngNormal(rng) * secondScale;
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

static Action SelectAction(GenDqn *dqn, int mazeIndex, int state, float epsilon, Rng *rng)
{
    if (RngFloat(rng) < epsilon) return (Action)RngRange(rng, GEN_ACTIONS);
    float input[GEN_MAX_INPUT];
    float hidden[GEN_MAX_HIDDEN];
    float values[GEN_ACTIONS];
    Encode(dqn, mazeIndex, state, input);
    Forward(dqn, dqn->online, input, hidden, values);
    Action best[GEN_ACTIONS];
    int count = 1;
    float maximum = values[0];
    best[0] = ACTION_UP;
    for (int action = 1; action < GEN_ACTIONS; action++) {
        if (values[action] > maximum) {
            maximum = values[action];
            count = 1;
            best[0] = (Action)action;
        } else if (values[action] == maximum) {
            best[count++] = (Action)action;
        }
    }
    return best[RngRange(rng, count)];
}

static GenTransition TakeStep(const ExperimentMaze *maze, int mazeIndex, int state, Action action)
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
        mazeIndex,
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

static void TrainBatch(GenDqn *dqn, Rng *rng)
{
    memset(dqn->gradient, 0, sizeof(float) * (size_t)dqn->parameterCount);
    int b1 = B1Offset(dqn);
    int w2 = W2Offset(dqn);
    int b2 = B2Offset(dqn);
    for (int sample = 0; sample < GEN_BATCH_SIZE; sample++) {
        GenTransition transition = dqn->replay.entries[RngRange(rng, dqn->replay.count)];
        float input[GEN_MAX_INPUT];
        float hidden[GEN_MAX_HIDDEN];
        float values[GEN_ACTIONS];
        Encode(dqn, transition.mazeIndex, transition.state, input);
        Forward(dqn, dqn->online, input, hidden, values);

        float target = transition.reward;
        if (!transition.done) {
            float nextInput[GEN_MAX_INPUT];
            float nextHidden[GEN_MAX_HIDDEN];
            float nextValues[GEN_ACTIONS];
            Encode(dqn, transition.mazeIndex, transition.nextState, nextInput);
            Forward(dqn, dqn->target, nextInput, nextHidden, nextValues);
            target += 0.95f * Maximum(nextValues);
        }
        float error = values[transition.action] - target;
        float outputGradient = (error > 1.0f ? 1.0f : error < -1.0f ? -1.0f : error) /
            GEN_BATCH_SIZE;
        dqn->gradient[b2 + transition.action] += outputGradient;
        int outputRow = w2 + transition.action * dqn->hiddenSize;
        for (int h = 0; h < dqn->hiddenSize; h++) {
            dqn->gradient[outputRow + h] += outputGradient * hidden[h];
            if (hidden[h] > 0.0f) {
                float hiddenGradient = outputGradient * dqn->online[outputRow + h];
                dqn->gradient[b1 + h] += hiddenGradient;
                int inputRow = h * dqn->inputSize;
                for (int i = 0; i < dqn->inputSize; i++)
                    dqn->gradient[inputRow + i] += hiddenGradient * input[i];
            }
        }
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

static GenEpisode RunEpisode(
    GenDqn *dqn,
    int mazeIndex,
    float epsilon,
    bool learn,
    Rng *rng)
{
    const ExperimentMaze *maze = &dqn->mazes[mazeIndex];
    GenEpisode result = {0};
    int state = maze->startState;
    for (int step = 0; step < GEN_MAX_STEPS; step++) {
        Action action = SelectAction(dqn, mazeIndex, state, epsilon, rng);
        GenTransition transition = TakeStep(maze, mazeIndex, state, action);
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
    return (GenOptions){5000, 3, 1, "generalization.csv"};
}

static bool ParsePositive(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > 1000000) return false;
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
        } else return false;
    }
    return true;
}

static const char *ObservationName(ObservationKind kind)
{
    return kind == OBS_LAYOUT ? "layout_aware" : "position_only";
}

static int RunOne(
    FILE *csv,
    const ExperimentMaze mazes[GEN_MAZE_COUNT],
    ObservationKind observation,
    int episodes,
    uint64_t seed)
{
    Rng rng;
    RngSeed(&rng, seed ^ ((uint64_t)observation << 48));
    GenDqn dqn;
    if (!InitializeDqn(&dqn, observation, mazes, &rng)) {
        DestroyDqn(&dqn);
        return 1;
    }
    float epsilon = 1.0f;
    clock_t start = clock();
    for (int episode = 0; episode < episodes; episode++) {
        int mazeIndex = RngRange(&rng, GEN_TRAIN_MAZES);
        RunEpisode(&dqn, mazeIndex, epsilon, true, &rng);
        epsilon *= 0.9995f;
        if (epsilon < 0.05f) epsilon = 0.05f;
    }
    double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;

    int groupSuccess[4] = {0};
    int groupTotal[4] = {0};
    for (int mazeIndex = 0; mazeIndex < GEN_MAZE_COUNT; mazeIndex++) {
        Rng evaluationRng;
        RngSeed(&evaluationRng, seed ^ (uint64_t)(mazeIndex + 1) ^ UINT64_C(0x6a09e667));
        GenEpisode evaluation = RunEpisode(&dqn, mazeIndex, 0.0f, false, &evaluationRng);
        const ExperimentMaze *maze = &mazes[mazeIndex];
        int gap = evaluation.reachedGoal ? evaluation.steps - maze->optimalSteps : -1;
        fprintf(csv, "%llu,%s,%s,%s,%llu,%dx%d,%d,%d,%d,%d,%.1f,%.3f,%d\n",
            (unsigned long long)seed,
            ObservationName(observation),
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
    printf("seed=%llu observation=%s train=%d/%d same=%d/%d smaller=%d/%d larger=%d/%d time=%.0fms\n",
        (unsigned long long)seed,
        ObservationName(observation),
        groupSuccess[0], groupTotal[0], groupSuccess[1], groupTotal[1],
        groupSuccess[2], groupTotal[2], groupSuccess[3], groupTotal[3], elapsed);
    DestroyDqn(&dqn);
    return 0;
}

int RunGeneralizationExperiment(int argc, char **argv)
{
    GenOptions options;
    if (!ParseOptions(argc, argv, &options)) {
        fprintf(stderr, "Usage: %s --generalization [--episodes N] [--seeds N] [--seed N] [--csv FILE]\n", argv[0]);
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
    fprintf(csv, "seed,observation,split,maze,maze_seed,size,optimal_steps,success,steps,excess_steps,return,training_ms,parameters\n");
    int status = 0;
    for (int offset = 0; offset < options.seeds && status == 0; offset++) {
        uint64_t seed = options.firstSeed + (uint64_t)offset;
        status = RunOne(csv, mazes, OBS_POSITION, options.episodes, seed);
        if (status == 0) status = RunOne(csv, mazes, OBS_LAYOUT, options.episodes, seed);
        fflush(csv);
    }
    fclose(csv);
    if (status == 0) printf("Wrote generalization results to %s\n", options.csvPath);
    return status;
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
    GenTransition wall = TakeStep(&mazes[0], 0, mazes[0].startState, ACTION_UP);
    if (wall.nextState != mazes[0].startState || wall.reward != -5.0f || wall.done)
        return false;
    if (GEN_LAYOUT_INPUT != 340 || ParameterCount(GEN_LAYOUT_INPUT, 64) != 22084)
        return false;
    return true;
}
