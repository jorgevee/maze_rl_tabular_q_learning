#include "raylib.h"
#include "maze.h"
#include "agent.h"

int main(void)
{
    InitWindow(800, 800, "Q-Learning Maze");

    InitializeAgent();
    InitializeQTable();

    SetTargetFPS(60);

    // Hyperparameters for Q-learning
    const float alpha = 0.1f;
    const float gamma = 0.95f;

    float epsilon = 1.0f;
    const float epsilonMin = 0.05f;
    const float epsilonDecay = 0.995f;

    const int trainingEpisodes = 5000;
    const int episodesPerFrame = 50;

    int episodesTrained = 0;
    float lastReward = 0.0f;
    bool training = false;

    // Order matters: draw the maze first, then the agent on top.
    while (!WindowShouldClose())
    {
        // Update the agent from keyboard input
        if (IsKeyPressed(KEY_UP))
        {
            TryMove(&agentPosition, ACTION_UP);
        }
        else if (IsKeyPressed(KEY_RIGHT))
        {
            TryMove(&agentPosition, ACTION_RIGHT);
        }
        else if (IsKeyPressed(KEY_DOWN))
        {
            TryMove(&agentPosition, ACTION_DOWN);
        }
        else if (IsKeyPressed(KEY_LEFT))
        {
            TryMove(&agentPosition, ACTION_LEFT);
        }

        // Toggle training
        if (IsKeyPressed(KEY_T))
        {
            training = !training;
        }

        // Train Q-learning episodes
        if (training)
        {
            for (
                int episode = 0;
                episode < episodesPerFrame &&
                episodesTrained < trainingEpisodes;
                episode++
            )
            {
                lastReward = TrainEpisode(
                    epsilon,
                    alpha,
                    gamma
                );

                epsilon *= epsilonDecay;

                if (epsilon < epsilonMin)
                {
                    epsilon = epsilonMin;
                }

                episodesTrained++;

                if (episodesTrained % 100 == 0)
                {
                    TraceLog(
                        LOG_INFO,
                        "Episode: %d | Reward: %.1f | Epsilon: %.3f",
                        episodesTrained,
                        lastReward,
                        epsilon
                    );
                }
            }

            if (episodesTrained >= trainingEpisodes)
            {
                training = false;

                TraceLog(
                    LOG_INFO,
                    "Training complete after %d episodes",
                    episodesTrained
                );
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawText("Maze RL", 20, 20, 30, BLACK);

        DrawMaze();
        DrawAgent(agentPosition);

        // Display the current state
        int currentState = PositionToState(agentPosition);

        DrawText(
            TextFormat("Current State: %d", currentState),
            200,
            10,
            18,
            BLACK
        );

        DrawText(
            TextFormat("Epsilon: %.3f", epsilon),
            200,
            35,
            18,
            BLACK
        );

        DrawText(
            TextFormat(
                "Episodes: %d / %d",
                episodesTrained,
                trainingEpisodes
            ),
            430,
            10,
            18,
            BLACK
        );

        DrawText(
            training
                ? "Training: RUNNING"
                : "Training: PAUSED",
            430,
            35,
            18,
            training ? RED : DARKGRAY
        );

        EndDrawing();
    }

    CloseWindow();

    return 0;
}