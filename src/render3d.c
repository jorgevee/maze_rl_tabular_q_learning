#include "render3d.h"
#include "mazesuite.h"
#include "ppo.h"
#include <stdint.h>
#include "raylib.h"
#include "raymath.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R3D_DEFAULT_WIDTH 1280
#define R3D_DEFAULT_HEIGHT 720
#define R3D_CELL 1.0f
#define R3D_WALL_HEIGHT 0.85f
#define R3D_STEPS_PER_SECOND 4.0f

/* Palette chosen to read clearly at a glance: cool grey walls, warm pale
   floor, a single saturated green for the goal so it is findable anywhere
   in frame. */
static const Color WALL_TOP = {126, 134, 148, 255};
static const Color WALL_SIDE = {84, 92, 106, 255};
static const Color WALL_EDGE = {58, 65, 78, 255};
static const Color FLOOR_LIGHT = {222, 220, 214, 255};
static const Color FLOOR_DARK = {203, 201, 195, 255};
static const Color FLOOR_LINE = {176, 175, 170, 255};
static const Color GOAL_GREEN = {46, 214, 96, 255};
static const Color AGENT_BODY = {238, 241, 245, 255};
static const Color AGENT_MARK = {40, 130, 240, 255};
static const Color SKY_TOP = {126, 172, 224, 255};
static const Color SKY_BOTTOM = {198, 220, 240, 255};
static const Color PANEL_BG = {18, 24, 34, 214};
static const Color PANEL_LINE = {96, 112, 136, 255};

typedef struct {
    int mazeIndex;
    int width;
    int height;
    bool paused;
    const char *policyPath;
    bool train;
    int minSeparation;
    uint64_t seed;
    const char *framesPath;   /* set => render frames for video instead of a window */
} Render3DOptions;

/* World position of a maze cell. The grid is centered on the origin so the
   camera framing does not depend on maze size (the suite mixes 8x8, 10x10
   and 12x12). */
static Vector3 CellToWorld(const ExperimentMaze *maze, int x, int y, float height)
{
    return (Vector3){
        ((float)x - (float)(maze->width - 1) * 0.5f) * R3D_CELL,
        height,
        ((float)y - (float)(maze->height - 1) * 0.5f) * R3D_CELL
    };
}

/* Breadth-first route, not just its length: the renderer needs the actual
   cells to walk. Returns the number of states written, or 0 if unreachable. */
static int ShortestRoute(const ExperimentMaze *maze, int route[GEN_MAX_CELLS])
{
    int parent[GEN_MAX_CELLS];
    int queue[GEN_MAX_CELLS];
    for (int i = 0; i < GEN_MAX_CELLS; i++) parent[i] = -2;
    static const int dx[GEN_ACTIONS] = {0, 1, 0, -1};
    static const int dy[GEN_ACTIONS] = {-1, 0, 1, 0};

    int front = 0, back = 0;
    queue[back++] = maze->startState;
    parent[maze->startState] = -1;
    bool found = false;
    while (front < back && !found) {
        int state = queue[front++];
        if (state == maze->goalState) { found = true; break; }
        int x = StateX(state), y = StateY(state);
        for (int a = 0; a < GEN_ACTIONS; a++) {
            int nx = x + dx[a], ny = y + dy[a];
            if (nx < 0 || nx >= maze->width || ny < 0 || ny >= maze->height) continue;
            int next = StateOf(nx, ny);
            if (maze->wall[next] || parent[next] != -2) continue;
            parent[next] = state;
            queue[back++] = next;
        }
    }
    if (!found) return 0;

    int reversed[GEN_MAX_CELLS];
    int count = 0;
    for (int s = maze->goalState; s != -1; s = parent[s]) reversed[count++] = s;
    for (int i = 0; i < count; i++) route[i] = reversed[count - 1 - i];
    return count;
}

/* Draws text truncated with an ellipsis if it would exceed maxWidth, so
   left- and right-aligned header blocks cannot collide at any resolution. */
static void DrawTextClipped(const char *text, int x, int y, int size, Color color, int maxWidth)
{
    if (MeasureText(text, size) <= maxWidth) {
        DrawText(text, x, y, size, color);
        return;
    }
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s", text);
    int length = (int)strlen(buffer);
    while (length > 3 && MeasureText(buffer, size) > maxWidth) {
        buffer[--length] = '\0';
        if (length > 3) { buffer[length - 1] = '.'; buffer[length - 2] = '.'; }
    }
    DrawText(buffer, x, y, size, color);
}

static void DrawSky(int width, int height)
{
    DrawRectangleGradientV(0, 0, width, height, SKY_TOP, SKY_BOTTOM);
}

static void DrawFloor(const ExperimentMaze *maze)
{
    for (int y = 0; y < maze->height; y++) {
        for (int x = 0; x < maze->width; x++) {
            Vector3 position = CellToWorld(maze, x, y, -0.02f);
            /* Subtle checker so the grid reads as tiles rather than a slab. */
            Color tile = ((x + y) % 2 == 0) ? FLOOR_LIGHT : FLOOR_DARK;
            DrawCube(position, R3D_CELL, 0.04f, R3D_CELL, tile);
            DrawCubeWires(position, R3D_CELL, 0.04f, R3D_CELL, FLOOR_LINE);
        }
    }
}

/* DrawCube shades every face identically, so a plain cube reads as a flat
   silhouette. Drawing the top separately in a lighter tone gives the walls
   enough form to be readable without pulling in a lighting shader. */
static void DrawWallBlock(Vector3 base)
{
    Vector3 body = {base.x, R3D_WALL_HEIGHT * 0.5f, base.z};
    DrawCube(body, R3D_CELL, R3D_WALL_HEIGHT, R3D_CELL, WALL_SIDE);
    Vector3 cap = {base.x, R3D_WALL_HEIGHT - 0.02f, base.z};
    DrawCube(cap, R3D_CELL, 0.04f, R3D_CELL, WALL_TOP);
    DrawCubeWires(body, R3D_CELL, R3D_WALL_HEIGHT, R3D_CELL, WALL_EDGE);
}

static void DrawWalls(const ExperimentMaze *maze)
{
    for (int y = 0; y < maze->height; y++) {
        for (int x = 0; x < maze->width; x++) {
            if (!maze->wall[StateOf(x, y)]) continue;
            DrawWallBlock(CellToWorld(maze, x, y, 0.0f));
        }
    }
}

static void DrawGoal(const ExperimentMaze *maze, double time)
{
    int gx = StateX(maze->goalState), gy = StateY(maze->goalState);
    /* Gentle pulse so the goal is obvious even when partly occluded. */
    float pulse = 0.5f + 0.5f * (float)((sin(time * 2.5) + 1.0) * 0.5);
    float size = 0.42f + 0.03f * pulse;
    Vector3 position = CellToWorld(maze, gx, gy, 0.30f);
    DrawCube(position, size, size, size, GOAL_GREEN);
    DrawCubeWires(position, size + 0.06f, size + 0.06f, size + 0.06f,
        (Color){180, 255, 200, (unsigned char)(90 + 90 * pulse)});
}

static void DrawAgent(Vector3 position)
{
    DrawCube(position, 0.46f, 0.34f, 0.46f, AGENT_BODY);
    DrawCubeWires(position, 0.46f, 0.34f, 0.46f, (Color){150, 160, 175, 255});
    /* Front stripe, so facing/motion is legible while it moves. */
    Vector3 mark = {position.x, position.y + 0.02f, position.z + 0.22f};
    DrawCube(mark, 0.10f, 0.20f, 0.03f, AGENT_MARK);
}

static void DrawTrail(const ExperimentMaze *maze, const int *route, int visited)
{
    for (int i = 0; i < visited; i++) {
        int x = StateX(route[i]), y = StateY(route[i]);
        Vector3 position = CellToWorld(maze, x, y, 0.03f);
        unsigned char fade = (unsigned char)(70 + 120 * ((float)(i + 1) / (float)visited));
        DrawCube(position, 0.24f, 0.02f, 0.24f, (Color){255, 165, 70, fade});
    }
}

static void DrawInfoPanel(
    const ExperimentMaze *maze,
    int step,
    float reward,
    int state,
    bool done,
    bool paused,
    bool fromPolicy,
    bool failed)
{
    const int x = 18, y = 18, w = 306, h = 150;
    DrawRectangle(x, y, w, h, PANEL_BG);
    DrawRectangleLines(x, y, w, h, PANEL_LINE);
    DrawText("RL Maze Environment", x + 16, y + 14, 22, RAYWHITE);
    DrawText("Reach the green goal!", x + 16, y + 42, 15, (Color){168, 194, 224, 255});
    DrawLine(x + 16, y + 64, x + w - 16, y + 64, PANEL_LINE);

    char line[128];
    snprintf(line, sizeof(line), "Step:     %d", step);
    DrawText(line, x + 16, y + 76, 16, RAYWHITE);
    snprintf(line, sizeof(line), "Reward:   %.1f", (double)reward);
    DrawText(line, x + 16, y + 96, 16, RAYWHITE);
    snprintf(line, sizeof(line), "Position: (%d, %d)", StateX(state), StateY(state));
    DrawText(line, x + 16, y + 116, 16, RAYWHITE);
    snprintf(line, sizeof(line), "Done:     %s", done ? "True" : "False");
    DrawText(line, x + 168, y + 116, 16, done ? GOAL_GREEN : RAYWHITE);

    snprintf(line, sizeof(line), "%s  |  %dx%d  |  optimal %d  |  %s",
        maze->name, maze->width, maze->height, maze->optimalSteps,
        fromPolicy ? "trained policy" : "BFS optimal route");
    DrawText(line, x + 16, y + h + 8, 14, (Color){40, 52, 68, 255});
    if (failed)
        DrawText("policy did not reach the goal", x + 16, y + h + 26, 14,
            (Color){168, 54, 44, 255});
    if (paused) DrawText("PAUSED", x + 246, y + 14, 16, (Color){255, 196, 84, 255});
}

static void DrawLegend(int screenWidth)
{
    const int w = 178, h = 132, x = screenWidth - w - 18, y = 18;
    DrawRectangle(x, y, w, h, PANEL_BG);
    DrawRectangleLines(x, y, w, h, PANEL_LINE);
    const char *labels[4] = {"Agent", "Goal", "Wall", "Floor"};
    Color swatches[4] = {AGENT_BODY, GOAL_GREEN, WALL_SIDE, FLOOR_LIGHT};
    for (int i = 0; i < 4; i++) {
        int row = y + 16 + i * 28;
        DrawRectangle(x + 18, row, 18, 18, swatches[i]);
        DrawRectangleLines(x + 18, row, 18, 18, PANEL_LINE);
        DrawText(labels[i], x + 48, row + 2, 16, RAYWHITE);
    }
}

static void DrawControls(int screenWidth, int screenHeight)
{
    const char *help = "SPACE pause   R restart   N/P maze   arrows orbit   ESC quit";
    int fontSize = 14;
    int textWidth = MeasureText(help, fontSize);
    DrawText(help, (screenWidth - textWidth) / 2, screenHeight - 26, fontSize,
        (Color){34, 46, 62, 220});
}


/* ---------- metric plots ----------

   raylib has no charting, so this is a small line-series widget: a ring
   buffer per series, shared auto-scaled axes, and a legend. Kept deliberately
   plain -- the point is reading trends, not decoration. */

#define PLOT_CAPACITY 512

typedef struct {
    float values[PLOT_CAPACITY];
    int count;
    int head;
} Series;

static void SeriesPush(Series *series, float value)
{
    series->values[series->head] = value;
    series->head = (series->head + 1) % PLOT_CAPACITY;
    if (series->count < PLOT_CAPACITY) series->count++;
}

static float SeriesAt(const Series *series, int index)
{
    int start = (series->head - series->count + PLOT_CAPACITY) % PLOT_CAPACITY;
    return series->values[(start + index) % PLOT_CAPACITY];
}

static float SeriesLast(const Series *series)
{
    return series->count ? SeriesAt(series, series->count - 1) : 0.0f;
}

/* Mean of the most recent `window` samples: the moving average that makes a
   noisy training-return curve readable. */
static float SeriesMovingAverage(const Series *series, int window)
{
    if (series->count == 0) return 0.0f;
    if (window > series->count) window = series->count;
    double sum = 0.0;
    for (int i = series->count - window; i < series->count; i++) sum += SeriesAt(series, i);
    return (float)(sum / (double)window);
}

typedef struct {
    const Series *series;
    Color color;
    const char *label;
} PlotSeries;

static void DrawPlot(
    Rectangle area,
    const char *title,
    const PlotSeries *series,
    int seriesCount,
    bool anchorZero,
    const char *valueFormat)
{
    DrawRectangleRec(area, (Color){22, 28, 39, 235});
    DrawRectangleLinesEx(area, 1.0f, (Color){62, 74, 92, 255});
    DrawText(title, (int)area.x + 10, (int)area.y + 7, 14, (Color){206, 218, 234, 255});

    const float padTop = 42.0f, padBottom = 16.0f, padLeft = 48.0f, padRight = 8.0f;
    Rectangle plot = {
        area.x + padLeft, area.y + padTop,
        area.width - padLeft - padRight, area.height - padTop - padBottom
    };

    float minimum = anchorZero ? 0.0f : 1e30f, maximum = -1e30f;
    int longest = 0;
    for (int s = 0; s < seriesCount; s++) {
        const Series *data = series[s].series;
        if (data->count > longest) longest = data->count;
        for (int i = 0; i < data->count; i++) {
            float v = SeriesAt(data, i);
            if (v < minimum) minimum = v;
            if (v > maximum) maximum = v;
        }
    }
    if (longest == 0) { DrawText("collecting...", (int)plot.x + 6, (int)(plot.y + plot.height * 0.45f), 13, (Color){110, 124, 145, 255}); return; }
    if (maximum <= minimum) maximum = minimum + 1.0f;
    float span = maximum - minimum;
    minimum -= span * 0.08f;
    maximum += span * 0.08f;
    if (anchorZero && minimum > 0.0f) minimum = 0.0f;

    /* Three gridlines with values, enough to read scale without clutter. */
    for (int g = 0; g <= 2; g++) {
        float t = (float)g / 2.0f;
        float y = plot.y + plot.height * (1.0f - t);
        DrawLine((int)plot.x, (int)y, (int)(plot.x + plot.width), (int)y,
            (Color){46, 56, 72, 255});
        char label[32];
        snprintf(label, sizeof(label), valueFormat, (double)(minimum + (maximum - minimum) * t));
        DrawText(label, (int)area.x + 6, (int)y - 6, 11, (Color){120, 136, 158, 255});
    }

    for (int s = 0; s < seriesCount; s++) {
        const Series *data = series[s].series;
        if (data->count < 2) continue;
        for (int i = 1; i < data->count; i++) {
            float x0 = plot.x + plot.width * ((float)(i - 1) / (float)(longest - 1));
            float x1 = plot.x + plot.width * ((float)i / (float)(longest - 1));
            float y0 = plot.y + plot.height *
                (1.0f - (SeriesAt(data, i - 1) - minimum) / (maximum - minimum));
            float y1 = plot.y + plot.height *
                (1.0f - (SeriesAt(data, i) - minimum) / (maximum - minimum));
            DrawLineEx((Vector2){x0, y0}, (Vector2){x1, y1}, 1.6f, series[s].color);
        }
    }

    /* Legend sits on its own row beneath the title: at narrow plot widths the
       two collided when sharing a line. */
    int legendX = (int)area.x + 10;
    for (int s = 0; s < seriesCount; s++) {
        char entry[64];
        snprintf(entry, sizeof(entry), "%s %.3g", series[s].label,
            (double)SeriesLast(series[s].series));
        int width = MeasureText(entry, 11);
        if (legendX + width > (int)(area.x + area.width) - 6) break;
        DrawText(entry, legendX, (int)area.y + 24, 11, series[s].color);
        legendX += width + 10;
    }
}

static bool ParseRender3DOptions(int argc, char **argv, Render3DOptions *options)
{
    options->mazeIndex = 0;
    options->width = R3D_DEFAULT_WIDTH;
    options->height = R3D_DEFAULT_HEIGHT;
    options->paused = false;
    options->policyPath = NULL;
    options->train = false;
    options->minSeparation = 10;
    options->seed = 1;
    options->framesPath = NULL;
    for (int i = 2; i < argc; i++) {
        char *end = NULL;
        if (strcmp(argv[i], "--maze") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 0 || v >= GEN_MAZE_COUNT) return false;
            options->mazeIndex = (int)v;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 640 || v > 3840) return false;
            options->width = (int)v;
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 360 || v > 2160) return false;
            options->height = (int)v;
        } else if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            options->policyPath = argv[++i];
        } else if (strcmp(argv[i], "--train") == 0) {
            options->train = true;
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            options->framesPath = argv[++i];
        } else if (strcmp(argv[i], "--min-separation") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 0 || v > 1000) return false;
            options->minSeparation = (int)v;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 1) return false;
            options->seed = (uint64_t)v;
        } else return false;
    }
    return true;
}

/* Rolls out a trained policy greedily, recording the cells it visits. Unlike
   the BFS route this can revisit cells, bump walls, or never finish -- which
   is the point: it shows what the policy actually does, including failing. */
static int PolicyRoute(
    const ExperimentMaze *maze,
    const PpoPolicy *policy,
    int route[GEN_MAX_STEPS + 1],
    bool *reachedGoal)
{
    GenMazeView view = ViewOfMaze(maze);
    float observation[GEN_LAYOUT_INPUT];
    int state = maze->startState;
    int count = 0;
    route[count++] = state;
    *reachedGoal = false;
    for (int step = 0; step < GEN_MAX_STEPS; step++) {
        EncodeLayout(&view, state, observation);
        int action = PpoPolicyGreedyAction(policy, observation);
        GenStepOutcome outcome = StepMaze(&view, state, (Action)action);
        state = outcome.nextState;
        route[count++] = state;
        if (outcome.done) { *reachedGoal = true; break; }
    }
    return count;
}


/* ---------- live training dashboard ----------

   Trains PPO inside the window, one rollout per frame, and draws the metric
   history alongside the current greedy policy walking a maze. Rendering the
   plots costs nothing next to a rollout, so this trains at very nearly
   headless speed -- the earlier "rendering is ~190x slower" figure applies
   only to animating one environment step per frame, which this does not do. */

typedef struct {
    Series trainReturn;
    Series trainReturnAverage;
    Series evalReturn;
    Series approxKl;
    Series clipFraction;
    Series entropy;
    Series livelock;
} MetricHistory;

/* Draws one complete dashboard frame. Shared by the live window and the
   offline video renderer so there is exactly one layout to maintain. */
static void DrawDashboardFrame(
    const ExperimentMaze *mazes,
    int mazeIndex,
    const MetricHistory *history,
    PpoTrainerMetrics m,
    PpoEvalSummary eval,
    const int *route,
    int routeLength,
    int animStep,
    bool routeSolved,
    Camera3D camera,
    int screenW,
    int screenH,
    const char *caption,
    const char *subcaption,
    bool training)
{
    const ExperimentMaze *maze = &mazes[mazeIndex];
    int panelH = screenH * 27 / 100;
    if (panelH < 180) panelH = 180;

    ClearBackground(SKY_BOTTOM);
    DrawSky(screenW, screenH - panelH);

    BeginMode3D(camera);
    DrawFloor(maze);
    DrawWalls(maze);
    if (routeLength > 0) DrawTrail(maze, route, animStep + 1);
    DrawGoal(maze, GetTime());
    if (routeLength > 0) {
        int state = route[animStep < routeLength ? animStep : routeLength - 1];
        DrawAgent(CellToWorld(maze, StateX(state), StateY(state), 0.19f));
    }
    EndMode3D();

    /* Header */
    int headerH = screenH * 11 / 100;
    if (headerH < 74) headerH = 74;
    DrawRectangle(0, 0, screenW, headerH, PANEL_BG);
    int titleSize = screenH / 34, bodySize = screenH / 46;
    if (titleSize < 18) titleSize = 18;
    if (bodySize < 14) bodySize = 14;

    char rightTop[160], rightBottom[160];
    bool heldOut = mazeIndex >= GEN_TRAIN_MAZES;
    snprintf(rightTop, sizeof(rightTop), "%s   %s", maze->name,
        heldOut ? "HELD-OUT (never trained on)" : "training maze");
    snprintf(rightBottom, sizeof(rightBottom),
        "steps %d   solved %d/%d   livelocked %d",
        m.environmentSteps, eval.solved, eval.total, eval.livelocked);

    int rightWidth = MeasureText(rightTop, bodySize);
    int otherWidth = MeasureText(rightBottom, bodySize);
    if (otherWidth > rightWidth) rightWidth = otherWidth;
    int leftLimit = screenW - rightWidth - 56;
    int row1 = headerH / 6, row2 = row1 + titleSize + 8;

    DrawTextClipped(caption, 22, row1, titleSize, RAYWHITE, leftLimit);
    if (subcaption)
        DrawTextClipped(subcaption, 22, row2, bodySize,
            (Color){168, 194, 224, 255}, leftLimit);

    DrawText(rightTop, screenW - MeasureText(rightTop, bodySize) - 22, row1, bodySize,
        heldOut ? (Color){255, 206, 120, 255} : (Color){168, 194, 224, 255});
    DrawText(rightBottom, screenW - MeasureText(rightBottom, bodySize) - 22,
        row2 + 2, bodySize, (Color){150, 168, 192, 255});

    if (routeLength > 0 && !training) {
        const char *verdict = routeSolved ? "SOLVED" : "did not reach the goal";
        Color verdictColor = routeSolved ? GOAL_GREEN : (Color){236, 116, 104, 255};
        int size = screenH / 30;
        int verdictWidth = MeasureText(verdict, size);
        int vx = (screenW - verdictWidth) / 2, vy = screenH - panelH - size - 22;
        /* Over the maze this had no contrast at all, so it gets its own plate. */
        DrawRectangleRounded(
            (Rectangle){(float)vx - 18.0f, (float)vy - 10.0f,
                        (float)verdictWidth + 36.0f, (float)size + 20.0f},
            0.35f, 8, (Color){14, 18, 28, 215});
        DrawText(verdict, vx, vy, size, verdictColor);
    }

    /* Metric strip */
    DrawRectangle(0, screenH - panelH, screenW, panelH, (Color){14, 19, 27, 245});
    DrawLine(0, screenH - panelH, screenW, screenH - panelH, PANEL_LINE);

    float gap = (float)screenW * 0.007f;
    float plotW = ((float)screenW - gap * 6.0f) / 5.0f;
    float plotY = (float)(screenH - panelH) + gap;
    float plotH = (float)panelH - gap * 2.0f;

    PlotSeries returns[3] = {
        {&history->trainReturn, (Color){88, 118, 168, 255}, "train"},
        {&history->trainReturnAverage, (Color){96, 190, 255, 255}, "avg"},
        {&history->evalReturn, (Color){86, 220, 130, 255}, "eval"}
    };
    DrawPlot((Rectangle){gap, plotY, plotW, plotH},
        "Episodic return", returns, 3, false, "%.0f");

    PlotSeries kl[1] = {{&history->approxKl, (Color){255, 150, 90, 255}, "KL"}};
    DrawPlot((Rectangle){gap * 2 + plotW, plotY, plotW, plotH},
        "Approx KL", kl, 1, true, "%.4f");

    PlotSeries clip[1] = {{&history->clipFraction, (Color){236, 130, 200, 255}, "frac"}};
    DrawPlot((Rectangle){gap * 3 + plotW * 2, plotY, plotW, plotH},
        "Clip fraction", clip, 1, true, "%.3f");

    PlotSeries entropy[1] = {{&history->entropy, (Color){214, 196, 108, 255}, "H"}};
    DrawPlot((Rectangle){gap * 4 + plotW * 3, plotY, plotW, plotH},
        "Policy entropy", entropy, 1, true, "%.2f");

    PlotSeries lock[1] = {{&history->livelock, (Color){232, 96, 96, 255}, "rate"}};
    DrawPlot((Rectangle){gap * 5 + plotW * 4, plotY, plotW, plotH},
        "Livelock rate", lock, 1, true, "%.2f");
}

/* The 3D scene renders across the whole window, but the header and the metric
   strip cover the top and bottom. `verticalShift` raises the look-at point so
   the maze sits centered in the band that is actually visible instead of being
   clipped by the panel. */
static Camera3D FrameCamera(const ExperimentMaze *maze, float orbit, float verticalShift)
{
    Camera3D camera = {0};
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float radius = 10.0f + (float)maze->width * 0.36f;
    camera.position = (Vector3){
        sinf(orbit) * radius,
        9.5f + (float)maze->height * 0.34f,
        cosf(orbit) * radius
    };
    camera.target = (Vector3){0.0f, verticalShift, 0.0f};
    return camera;
}

/* The scene renders across the whole window but only the band between the
   header and the metric strip is visible. Lowering the look-at point raises
   the scene on screen, so the shift is negative when the band sits above
   center -- which it always does here, the strip being taller than the
   header. */
static float BandShift(int screenH, int headerH, int panelH)
{
    float visibleCenter = (float)(headerH + (screenH - panelH)) * 0.5f;
    float screenCenter = (float)screenH * 0.5f;
    return (visibleCenter - screenCenter) * 0.020f;
}

static void PushMetrics(MetricHistory *history, PpoTrainerMetrics m, PpoEvalSummary eval)
{
    SeriesPush(&history->trainReturn, m.trainReturn);
    SeriesPush(&history->trainReturnAverage,
        SeriesMovingAverage(&history->trainReturn, 20));
    SeriesPush(&history->approxKl, m.approxKl);
    SeriesPush(&history->clipFraction, m.clipFraction);
    SeriesPush(&history->entropy, m.entropy);
    SeriesPush(&history->evalReturn, eval.meanReturn);
    SeriesPush(&history->livelock,
        eval.total > 0 ? (float)eval.livelocked / (float)eval.total : 0.0f);
}

static int RunTrainDashboard(const Render3DOptions *options, uint64_t seed, int minSeparation)
{
    PpoTrainer *trainer = PpoTrainerCreate(seed, true, minSeparation);
    if (!trainer) { fprintf(stderr, "Could not create trainer.\n"); return 1; }
    ExperimentMaze *mazes = malloc(sizeof(ExperimentMaze) * GEN_MAZE_COUNT);
    if (!mazes) { PpoTrainerDestroy(trainer); return 1; }
    BuildMazeSuite(mazes, GEN_SUITE_SEED);

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(options->width, options->height, "Maze RL - PPO live training");
    SetTargetFPS(60);

    MetricHistory history = {0};
    PpoEvalSummary eval = {0};
    int mazeIndex = options->mazeIndex;
    int route[GEN_MAX_STEPS + 2];
    int routeLength = 0;
    bool routeSolved = false;
    int animStep = 0;
    double animAccumulator = 0.0;
    bool training = true;
    float orbit = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) training = !training;
        if (IsKeyPressed(KEY_N)) mazeIndex = (mazeIndex + 1) % GEN_MAZE_COUNT;
        if (IsKeyPressed(KEY_P)) mazeIndex = (mazeIndex + GEN_MAZE_COUNT - 1) % GEN_MAZE_COUNT;
        if (IsKeyDown(KEY_LEFT)) orbit -= 0.9f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) orbit += 0.9f * GetFrameTime();

        if (training) {
            PpoTrainerStep(trainer);
            eval = PpoTrainerEvaluate(trainer, GEN_TRAIN_MAZES,
                GEN_MAZE_COUNT - GEN_TRAIN_MAZES);
            PushMetrics(&history, PpoTrainerLastMetrics(trainer), eval);
            routeLength = PpoTrainerGreedyRoute(trainer, mazeIndex, route,
                GEN_MAX_STEPS + 2, &routeSolved);
            animStep = 0;
        }

        if (routeLength > 1) {
            animAccumulator += GetFrameTime();
            if (animAccumulator >= 1.0 / 12.0) {
                animAccumulator = 0.0;
                animStep = (animStep + 1) % routeLength;
            }
        }

        BeginDrawing();
        DrawDashboardFrame(mazes, mazeIndex, &history, PpoTrainerLastMetrics(trainer),
            eval, route, routeLength, animStep, routeSolved,
            FrameCamera(&mazes[mazeIndex], orbit,
                BandShift(GetScreenHeight(), GetScreenHeight() * 11 / 100,
                          GetScreenHeight() * 27 / 100)),
            GetScreenWidth(), GetScreenHeight(),
            "PPO live training",
            training ? "SPACE pause   N/P maze   arrows orbit   ESC quit"
                     : "PAUSED   -   SPACE resume",
            training);
        EndDrawing();
    }

    CloseWindow();
    free(mazes);
    PpoTrainerDestroy(trainer);
    return 0;
}

/* ---------- offline video renderer ----------

   Writes numbered PNG frames for ffmpeg rather than screen-recording. That
   decouples training speed from video speed entirely: the boring middle of
   training can advance many rollouts per rendered frame, while the playback
   phases freeze the policy and animate one environment step at a time. The
   result is deterministic, any resolution, and regenerable after a code
   change. Mirrors the frame/manifest pipeline the DQN video already uses. */

typedef struct {
    const char *framesPath;
    int fps;
    int width;
    int height;
} VideoConfig;

static bool ExportFrame(RenderTexture2D target, const char *dir, int index)
{
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/frame_%05d.png", dir, index);
    if (n < 0 || (size_t)n >= sizeof(path)) return false;
    Image image = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&image);
    bool ok = ExportImage(image, path);
    UnloadImage(image);
    return ok;
}

static void DrawCard(int w, int h, const char *title, const char *subtitle, const char *footer)
{
    ClearBackground((Color){13, 18, 27, 255});
    int titleSize = h / 12, subSize = h / 30, footSize = h / 44;
    int tw = MeasureText(title, titleSize);
    DrawText(title, (w - tw) / 2, h * 34 / 100, titleSize, RAYWHITE);
    if (subtitle) {
        int sw = MeasureText(subtitle, subSize);
        DrawText(subtitle, (w - sw) / 2, h * 34 / 100 + titleSize + h / 40, subSize,
            (Color){120, 196, 255, 255});
    }
    if (footer) {
        int fw = MeasureText(footer, footSize);
        DrawText(footer, (w - fw) / 2, h * 78 / 100, footSize, (Color){130, 146, 168, 255});
    }
    DrawRectangle(w * 32 / 100, h * 62 / 100, w * 36 / 100, h / 180,
        (Color){70, 150, 230, 255});
}

static int RunVideoDirector(const Render3DOptions *options, uint64_t seed, int minSeparation)
{
    VideoConfig video = {options->framesPath, 30, options->width, options->height};
    if (!DirectoryExists(video.framesPath)) {
        fprintf(stderr, "Frame directory does not exist: %s\n", video.framesPath);
        return 1;
    }

    PpoTrainer *trainer = PpoTrainerCreate(seed, true, minSeparation);
    if (!trainer) return 1;
    ExperimentMaze *mazes = malloc(sizeof(ExperimentMaze) * GEN_MAZE_COUNT);
    if (!mazes) { PpoTrainerDestroy(trainer); return 1; }
    BuildMazeSuite(mazes, GEN_SUITE_SEED);

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN | FLAG_MSAA_4X_HINT);
    InitWindow(video.width, video.height, "Maze RL video");
    RenderTexture2D target = LoadRenderTexture(video.width, video.height);
    if (!IsRenderTextureValid(target)) {
        CloseWindow();
        free(mazes);
        PpoTrainerDestroy(trainer);
        return 1;
    }

    MetricHistory history = {0};
    PpoEvalSummary eval = {0};
    int route[GEN_MAX_STEPS + 2];
    int routeLength = 0;
    bool routeSolved = false;
    int frame = 0;
    bool ok = true;

    const int titleFrames = video.fps * 3;
    const int earlyFrames = video.fps * 8;
    const int compressedFrames = video.fps * 15;
    const int perMazeFrames = video.fps * 7 / 2;
    const int endFrames = video.fps * 4;
    /* Train to the same 575k-step budget the headline experiments use, spread
       over the compressed phase's screen time. An earlier version simply ran
       two rollouts per frame, which landed at ~1.9M steps -- far past the point
       where held-out solve rate peaks (see EXPERIMENTS.md), so the video showed
       the policy well after it had degraded. */
    const int rolloutSteps = 2048;
    const int totalUpdates = 575000 / rolloutSteps;
    const int earlyUpdates = earlyFrames / 8;
    const int compressedUpdates = totalUpdates - earlyUpdates;

    #define EMIT(caption, sub, mazeIdx, animIdx, isTraining) do { \
        BeginTextureMode(target); \
        DrawDashboardFrame(mazes, (mazeIdx), &history, PpoTrainerLastMetrics(trainer), \
            eval, route, routeLength, (animIdx), routeSolved, \
            FrameCamera(&mazes[mazeIdx], orbit, \
                BandShift(video.height, video.height * 11 / 100, \
                          video.height * 27 / 100)), \
            video.width, video.height, \
            (caption), (sub), (isTraining)); \
        EndTextureMode(); \
        ok = ExportFrame(target, video.framesPath, frame++) && ok; \
    } while (0)

    float orbit = 0.0f;

    /* Title */
    for (int i = 0; i < titleFrames && ok; i++) {
        BeginTextureMode(target);
        DrawCard(video.width, video.height,
            "Learning to Navigate Unseen Mazes",
            "PPO, written from scratch in C",
            "no machine-learning libraries  |  16 training mazes, 18 held out");
        EndTextureMode();
        ok = ExportFrame(target, video.framesPath, frame++) && ok;
    }

    /* Early training: slow, so the flailing and noisy metrics are visible. */
    int showMaze = GEN_TRAIN_MAZES;   /* a held-out maze throughout training */
    for (int i = 0; i < earlyFrames && ok; i++) {
        if (i % 8 == 0) {
            PpoTrainerStep(trainer);
            eval = PpoTrainerEvaluate(trainer, GEN_TRAIN_MAZES,
                GEN_MAZE_COUNT - GEN_TRAIN_MAZES);
            PushMetrics(&history, PpoTrainerLastMetrics(trainer), eval);
            routeLength = PpoTrainerGreedyRoute(trainer, showMaze, route,
                GEN_MAX_STEPS + 2, &routeSolved);
        }
        orbit += 0.0016f;
        EMIT("Start of training", "The policy is close to random - it explores, and mostly fails",
             showMaze, (i / 3) % (routeLength > 0 ? routeLength : 1), true);
    }

    /* Compressed training: many rollouts per frame. */
    int compressedDone = 0;
    for (int i = 0; i < compressedFrames && ok; i++) {
        int want = (int)(((long)(i + 1) * compressedUpdates) / compressedFrames);
        while (compressedDone < want) { PpoTrainerStep(trainer); compressedDone++; }
        /* A full held-out sweep is 18 rollouts; doing it every frame dominated
           render time for a curve that barely moves between frames. */
        if (i % 5 == 0)
            eval = PpoTrainerEvaluate(trainer, GEN_TRAIN_MAZES,
                GEN_MAZE_COUNT - GEN_TRAIN_MAZES);
        PushMetrics(&history, PpoTrainerLastMetrics(trainer), eval);
        routeLength = PpoTrainerGreedyRoute(trainer, showMaze, route,
            GEN_MAX_STEPS + 2, &routeSolved);
        orbit += 0.0016f;
        EMIT("Training", "Entropy falls, the early two-cell livelock resolves, held-out solves climb",
             showMaze, (i / 2) % (routeLength > 0 ? routeLength : 1), true);
    }

    /* Pick what to show: two training mazes it solves, then held-out mazes --
       including one it fails, shown deliberately rather than hidden. */
    int solvedTrain[2], solvedHeld[3], failedHeld = -1;
    int nTrain = 0, nHeld = 0;
    for (int i = 0; i < GEN_MAZE_COUNT; i++) {
        bool done = false;
        PpoTrainerGreedyRoute(trainer, i, route, GEN_MAX_STEPS + 2, &done);
        if (i < GEN_TRAIN_MAZES) { if (done && nTrain < 2) solvedTrain[nTrain++] = i; }
        else if (done) { if (nHeld < 3) solvedHeld[nHeld++] = i; }
        else if (failedHeld < 0) failedHeld = i;
    }

    /* Playback: policy frozen, one environment step per few frames. */
    for (int k = 0; k < nTrain && ok; k++) {
        routeLength = PpoTrainerGreedyRoute(trainer, solvedTrain[k], route,
            GEN_MAX_STEPS + 2, &routeSolved);
        for (int i = 0; i < perMazeFrames && ok; i++) {
            orbit += 0.0016f;
            int step = i / 3;
            if (step >= routeLength) step = routeLength - 1;
            EMIT("Trained policy: mazes it learned on", "Greedy rollout, no exploration",
                 solvedTrain[k], step, false);
        }
    }
    for (int k = 0; k < nHeld && ok; k++) {
        routeLength = PpoTrainerGreedyRoute(trainer, solvedHeld[k], route,
            GEN_MAX_STEPS + 2, &routeSolved);
        for (int i = 0; i < perMazeFrames && ok; i++) {
            orbit += 0.0016f;
            int step = i / 3;
            if (step >= routeLength) step = routeLength - 1;
            EMIT("Mazes it has never seen", "Different layout, different size - solved anyway",
                 solvedHeld[k], step, false);
        }
    }
    if (failedHeld >= 0 && ok) {
        routeLength = PpoTrainerGreedyRoute(trainer, failedHeld, route,
            GEN_MAX_STEPS + 2, &routeSolved);
        for (int i = 0; i < perMazeFrames && ok; i++) {
            orbit += 0.0016f;
            int step = i / 3;
            if (step >= routeLength) step = routeLength - 1;
            EMIT("...and mazes it still fails",
                 "It can see the goal but oscillates against a wall. Work in progress.",
                 failedHeld, step, false);
        }
    }

    /* End card */
    char summary[192];
    snprintf(summary, sizeof(summary), "this run: %d of %d unseen mazes solved",
        eval.solved, eval.total);
    for (int i = 0; i < endFrames && ok; i++) {
        BeginTextureMode(target);
        DrawCard(video.width, video.height, summary,
            "across 10 seeds: PPO 18.3% vs DQN 8.9% on held-out mazes",
            "tabular Q-learning  ->  DQN  ->  PPO   |   all hand-written in C");
        EndTextureMode();
        ok = ExportFrame(target, video.framesPath, frame++) && ok;
    }
    #undef EMIT

    char manifestPath[1024];
    snprintf(manifestPath, sizeof(manifestPath), "%s/manifest.txt", video.framesPath);
    FILE *manifest = fopen(manifestPath, "w");
    if (manifest) {
        fprintf(manifest, "fps=%d\nwidth=%d\nheight=%d\nframes=%d\n",
            video.fps, video.width, video.height, frame);
        fclose(manifest);
    }

    UnloadRenderTexture(target);
    CloseWindow();
    free(mazes);
    PpoTrainerDestroy(trainer);
    if (!ok) { fprintf(stderr, "Frame export failed.\n"); return 1; }
    printf("Wrote %d frames (%.1fs at %d fps) to %s\n",
        frame, (double)frame / video.fps, video.fps, video.framesPath);
    return 0;
}

int RunRender3D(int argc, char **argv)
{
    Render3DOptions options;
    if (!ParseRender3DOptions(argc, argv, &options)) {
        fprintf(stderr, "Usage: %s --render3d [--maze 0-%d] [--width N] [--height N] "
            "[--policy FILE] [--train [--seed N] [--min-separation N]] "
            "[--frames DIR]\n",
            argv[0], GEN_MAZE_COUNT - 1);
        return 2;
    }

    if (options.framesPath)
        return RunVideoDirector(&options, options.seed, options.minSeparation);
    if (options.train)
        return RunTrainDashboard(&options, options.seed, options.minSeparation);

    ExperimentMaze *mazes = malloc(sizeof(ExperimentMaze) * GEN_MAZE_COUNT);
    if (!mazes) return 1;
    BuildMazeSuite(mazes, GEN_SUITE_SEED);

    int mazeIndex = options.mazeIndex;
    PpoPolicy *policy = NULL;
    if (options.policyPath) {
        policy = PpoPolicyLoad(options.policyPath);
        if (!policy) {
            fprintf(stderr, "Could not load policy: %s\n", options.policyPath);
            free(mazes);
            return 1;
        }
    }

    int route[GEN_MAX_STEPS + 2];
    bool policySolved = false;
    int routeLength = policy
        ? PolicyRoute(&mazes[mazeIndex], policy, route, &policySolved)
        : ShortestRoute(&mazes[mazeIndex], route);

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(options.width, options.height, "Maze RL - 3D view");
    SetTargetFPS(60);

    /* Angled overhead framing: high enough to read the whole layout, low
       enough that the extruded walls still have visible sides. */
    Camera3D camera = {0};
    camera.position = (Vector3){0.0f, 11.0f, 11.5f};
    camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float orbit = 0.0f;

    int stepIndex = 0;
    float reward = 0.0f;
    double accumulator = 0.0;
    bool paused = options.paused;

    while (!WindowShouldClose()) {
        const ExperimentMaze *maze = &mazes[mazeIndex];

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { stepIndex = 0; reward = 0.0f; accumulator = 0.0; }
        if (IsKeyPressed(KEY_N) || IsKeyPressed(KEY_P)) {
            mazeIndex = IsKeyPressed(KEY_N)
                ? (mazeIndex + 1) % GEN_MAZE_COUNT
                : (mazeIndex + GEN_MAZE_COUNT - 1) % GEN_MAZE_COUNT;
            routeLength = policy
                ? PolicyRoute(&mazes[mazeIndex], policy, route, &policySolved)
                : ShortestRoute(&mazes[mazeIndex], route);
            stepIndex = 0; reward = 0.0f; accumulator = 0.0;
            maze = &mazes[mazeIndex];
        }
        if (IsKeyDown(KEY_LEFT)) orbit -= 0.9f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) orbit += 0.9f * GetFrameTime();

        float radius = 11.5f + (float)maze->width * 0.35f;
        camera.position = (Vector3){
            sinf(orbit) * radius,
            10.0f + (float)maze->height * 0.30f,
            cosf(orbit) * radius
        };

        /* Advance the walk on a wall-clock schedule so the animation speed
           does not depend on frame rate. */
        if (!paused && routeLength > 0 && stepIndex < routeLength - 1) {
            accumulator += GetFrameTime();
            if (accumulator >= 1.0 / R3D_STEPS_PER_SECOND) {
                accumulator = 0.0;
                int previous = route[stepIndex];
                stepIndex++;
                int current = route[stepIndex];
                if (current == maze->goalState) reward += 100.0f;
                else if (current == previous) reward += -5.0f;   /* bumped a wall */
                else reward += -1.0f;
            }
        }

        int state = routeLength > 0 ? route[stepIndex] : maze->startState;
        bool atEnd = routeLength > 0 && stepIndex == routeLength - 1;
        bool done = atEnd && (policy ? policySolved : true);

        BeginDrawing();
        ClearBackground(SKY_BOTTOM);
        DrawSky(GetScreenWidth(), GetScreenHeight());

        BeginMode3D(camera);
        DrawFloor(maze);
        DrawWalls(maze);
        if (routeLength > 0) DrawTrail(maze, route, stepIndex + 1);
        DrawGoal(maze, GetTime());
        DrawAgent(CellToWorld(maze, StateX(state), StateY(state), 0.19f));
        EndMode3D();

        DrawInfoPanel(maze, stepIndex, reward, state, done, paused,
            policy != NULL, atEnd && !done);
        DrawLegend(GetScreenWidth());
        DrawControls(GetScreenWidth(), GetScreenHeight());
        EndDrawing();
    }

    CloseWindow();
    if (policy) PpoPolicyDestroy(policy);
    free(mazes);
    return 0;
}
