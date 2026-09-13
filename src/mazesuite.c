#include "mazesuite.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Implementation of the shared maze environment. Moved verbatim out of
   generalization.c, where it had grown up alongside the DQN experiments; the
   logic is unchanged, and the DQN results were verified byte-for-byte
   identical across the move. */

int StateOf(int x, int y) { return y * GEN_MAX_SIZE + x; }
int StateX(int state) { return state % GEN_MAX_SIZE; }
int StateY(int state) { return state / GEN_MAX_SIZE; }

int ShortestPathGeneric(
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

int ShortestPath(const ExperimentMaze *maze)
{
    return ShortestPathGeneric(maze->width, maze->height, maze->wall,
        maze->startState, maze->goalState);
}

GenMazeView ViewOfMaze(const ExperimentMaze *maze)
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
void GenerateProceduralMaze(
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
void RandomizeStartGoal(
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

void BuildMazeSuite(ExperimentMaze mazes[GEN_MAZE_COUNT], uint64_t suiteSeed)
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

void EncodeLayout(const GenMazeView *maze, int state, float output[GEN_LAYOUT_INPUT])
{
    memset(output, 0, sizeof(float) * GEN_LAYOUT_INPUT);
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

GenStepOutcome StepMaze(const GenMazeView *maze, int state, Action action)
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
    return (GenStepOutcome){
        nextState,
        done ? 100.0f : blocked ? -5.0f : -1.0f,
        done
    };
}

