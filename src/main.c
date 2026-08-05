#include "raylib.h"
#include "maze.h"
#include "agent.h"
int main(void)
{
    InitWindow(800, 800, "Q-Learning Maze");
    InitializeAgent();
    SetTargetFPS(60);

    //  Order matters: the maze is drawn first, then the red agent is drawn on top of it.
    while (!WindowShouldClose())
    {
        // Update the agent from keyboard input
        if (IsKeyPressed(KEY_UP))
            TryMove(&agentPosition, ACTION_UP);
        else if (IsKeyPressed(KEY_RIGHT))
            TryMove(&agentPosition, ACTION_RIGHT);
        else if (IsKeyPressed(KEY_DOWN))
            TryMove(&agentPosition, ACTION_DOWN);
        else if (IsKeyPressed(KEY_LEFT))
            TryMove(&agentPosition, ACTION_LEFT);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawText("Maze RL", 20, 20, 30, BLACK);
        DrawMaze();
        DrawAgent(agentPosition);
        // Make the state visible
        int currentState = PositionToState(agentPosition);
        DrawText(TextFormat("Current State: %d", currentState), 200, 25, 20, BLACK);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}