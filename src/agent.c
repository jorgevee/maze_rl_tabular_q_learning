#include "agent.h"
#include "raylib.h"
#include <math.h>

Position startPosition = {0, 0};
Position agentPosition = {0, 0};
Position goalPosition = {0, 0};

void InitializeAgent(void)
{
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (maze[y][x] == CELL_START) startPosition = (Position){x, y};
            else if (maze[y][x] == CELL_GOAL) goalPosition = (Position){x, y};
        }
    }
    ResetAgent();
}

void ResetAgent(void) { agentPosition = startPosition; }

bool TryMove(Position *position, Action action)
{
    static const int dx[ACTION_COUNT] = {0, 1, 0, -1};
    static const int dy[ACTION_COUNT] = {-1, 0, 1, 0};
    int targetX = position->x + dx[action];
    int targetY = position->y + dy[action];
    if (targetX < 0 || targetX >= GRID_WIDTH || targetY < 0 || targetY >= GRID_HEIGHT)
        return false;
    if (maze[targetY][targetX] == CELL_WALL) return false;
    position->x = targetX;
    position->y = targetY;
    return true;
}

void DrawAgent(Position position)
{
    int centerX = MAZE_OFFSET_X + position.x * CELL_SIZE + CELL_SIZE / 2;
    int centerY = MAZE_OFFSET_Y + position.y * CELL_SIZE + CELL_SIZE / 2;
    DrawCircle(centerX, centerY, CELL_SIZE / 3, RED);
}

static bool PolicyAction(const Learner *learner, int state, Action *bestAction)
{
    float values[ACTION_COUNT];
    LearnerGetQValues(learner, state, values);
    float best = values[0];
    *bestAction = ACTION_UP;
    bool differ = false;
    for (int action = 1; action < ACTION_COUNT; action++) {
        if (fabsf(values[action] - values[0]) > 1.0e-6f) differ = true;
        if (values[action] > best) { best = values[action]; *bestAction = (Action)action; }
    }
    return differ;
}

void DrawPolicy(const Learner *learner)
{
    static const float dx[ACTION_COUNT] = {0.0f, 1.0f, 0.0f, -1.0f};
    static const float dy[ACTION_COUNT] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (maze[y][x] == CELL_WALL || maze[y][x] == CELL_GOAL) continue;
            int state = PositionToState((Position){x, y});
            Action action;
            if (!PolicyAction(learner, state, &action)) continue;
            Vector2 center = {MAZE_OFFSET_X + x * CELL_SIZE + CELL_SIZE / 2.0f,
                              MAZE_OFFSET_Y + y * CELL_SIZE + CELL_SIZE / 2.0f};
            Vector2 direction = {dx[action], dy[action]};
            Vector2 tip = {center.x + direction.x * 20.0f, center.y + direction.y * 20.0f};
            Vector2 base = {tip.x - direction.x * 8.0f, tip.y - direction.y * 8.0f};
            Vector2 perpendicular = {-direction.y, direction.x};
            Vector2 left = {base.x + perpendicular.x * 6.0f, base.y + perpendicular.y * 6.0f};
            Vector2 right = {base.x - perpendicular.x * 6.0f, base.y - perpendicular.y * 6.0f};
            DrawLineEx(center, tip, 3.0f, PURPLE);
            DrawLineEx(tip, left, 3.0f, PURPLE);
            DrawLineEx(tip, right, 3.0f, PURPLE);
        }
    }
}

void DrawValueHeatmap(const Learner *learner)
{
    float minimum = 0.0f;
    float maximum = 0.0f;
    bool found = false;
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (maze[y][x] == CELL_WALL || maze[y][x] == CELL_GOAL) continue;
            float value = LearnerMaximumQ(learner, PositionToState((Position){x, y}));
            if (!found) { minimum = maximum = value; found = true; }
            else { if (value < minimum) minimum = value; if (value > maximum) maximum = value; }
        }
    }
    if (!found || maximum <= minimum) return;
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (maze[y][x] == CELL_WALL || maze[y][x] == CELL_GOAL) continue;
            float value = LearnerMaximumQ(learner, PositionToState((Position){x, y}));
            float normalized = (value - minimum) / (maximum - minimum);
            Color color = {(unsigned char)(40.0f + 215.0f * normalized),
                           (unsigned char)(80.0f + 120.0f * normalized),
                           (unsigned char)(220.0f - 180.0f * normalized), 140};
            DrawRectangle(MAZE_OFFSET_X + x * CELL_SIZE + 1,
                          MAZE_OFFSET_Y + y * CELL_SIZE + 1,
                          CELL_SIZE - 2, CELL_SIZE - 2, color);
        }
    }
}
