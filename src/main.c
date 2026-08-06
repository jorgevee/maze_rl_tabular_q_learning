#include "raylib.h"
#include "maze.h"
#include "agent.h"
#include "environment.h"

#define METRIC_WINDOW 100
#define PREVIEW_MAX_STEPS 60

typedef struct
{
    bool active;
    int checkpoint;
    int state;
    int steps;
    float timer;
} PolicyPreview;

static void StartPolicyPreview(
    PolicyPreview *preview,
    int checkpoint
)
{
    preview->active = true;
    preview->checkpoint = checkpoint;
    preview->state = PositionToState(startPosition);
    preview->steps = 0;
    preview->timer = 0.0f;

    ResetAgent();

    TraceLog(
        LOG_INFO,
        "Starting greedy preview at episode %d",
        checkpoint
    );
}

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
    const int episodesPerFrame = 5;

    int episodesTrained = 0;
    bool training = false;

    // Metrics for recent performance
    bool recentSuccesses[METRIC_WINDOW] = { false };
    int recentSteps[METRIC_WINDOW] = { 0 };

    int recentCount = 0;

    float recentSuccessRate = 0.0f;
    float recentAverageSteps = 0.0f;

    EpisodeResult lastEpisode = { 0 };

    bool showPolicy = false;
    bool showValues = false;

    // Preview checkpoints
    const int previewCheckpoints[] =
    {
        0,
        100,
        500,
        1000,
        5000
    };

    const int previewCheckpointCount =
        sizeof(previewCheckpoints) /
        sizeof(previewCheckpoints[0]);

    int nextPreviewIndex = 0;

    PolicyPreview preview = { 0 };

    const float previewStepDelay = 0.10f;

    // Main window loop
    while (!WindowShouldClose())
    {
        // Manual movement
        if (!training && !preview.active)
  {
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
  }

        // Toggle training
        if (IsKeyPressed(KEY_T))
        {
            training = !training;

            if (!training)
            {
                preview.active = false;
            }
        }

        if (IsKeyPressed(KEY_P))
        {
            showPolicy = !showPolicy;
        }

        if (IsKeyPressed(KEY_V))
        {
            showValues = !showValues;
        }

        // Start preview at configured checkpoints
        if (
            training &&
            !preview.active &&
            nextPreviewIndex < previewCheckpointCount &&
            episodesTrained >=
                previewCheckpoints[nextPreviewIndex]
        )
        {
            StartPolicyPreview(
                &preview,
                previewCheckpoints[nextPreviewIndex]
            );

            nextPreviewIndex++;
        }

        // Train Q-learning episodes
        if (training && !preview.active)
        {
            for (
                int episode = 0;
                episode < episodesPerFrame &&
                episodesTrained < trainingEpisodes;
                episode++
            )
            {
                lastEpisode = TrainEpisode(
                    episodesTrained + 1,
                    epsilon,
                    alpha,
                    gamma
                );

                epsilon *= epsilonDecay;

                if (epsilon < epsilonMin)
                {
                    epsilon = epsilonMin;
                }

                int metricIndex =
                    episodesTrained % METRIC_WINDOW;

                recentSuccesses[metricIndex] =
                    lastEpisode.reachedGoal;

                recentSteps[metricIndex] =
                    lastEpisode.steps;

                if (recentCount < METRIC_WINDOW)
                {
                    recentCount++;
                }

                episodesTrained++;

                int successCount = 0;
                int successfulStepTotal = 0;

                for (
                    int index = 0;
                    index < recentCount;
                    index++
                )
                {
                    if (recentSuccesses[index])
                    {
                        successCount++;
                        successfulStepTotal +=
                            recentSteps[index];
                    }
                }

                if (recentCount > 0)
                {
                    recentSuccessRate =
                        100.0f *
                        successCount /
                        recentCount;
                }

                if (successCount > 0)
                {
                    recentAverageSteps =
                        (float)successfulStepTotal /
                        successCount;
                }
                else
                {
                    recentAverageSteps = 0.0f;
                }

                if (episodesTrained % 100 == 0)
                {
                    TraceLog(
                        LOG_INFO,
                        "Episode: %d | Reward: %.1f | Success Rate: %.2f%% | Avg Steps: %.2f | Epsilon: %.3f",
                        episodesTrained,
                        lastEpisode.totalReward,
                        recentSuccessRate,
                        recentAverageSteps,
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

            if (
                !preview.active &&
                nextPreviewIndex < previewCheckpointCount &&
                episodesTrained >=
                    previewCheckpoints[nextPreviewIndex]
            )
            {
                StartPolicyPreview(
                    &preview,
                    previewCheckpoints[nextPreviewIndex]
                );

                nextPreviewIndex++;
            }
        }
        if (preview.active)
  {
      preview.timer += GetFrameTime();

      if (preview.timer >= previewStepDelay)
      {
          preview.timer = 0.0f;

          Action action =
              ChooseAction(preview.state, 0.0f);

          StepResult result =
              EnvironmentStep(
                  preview.state,
                  action
              );

          preview.state = result.nextState;

          agentPosition =
              StateToPosition(preview.state);

          preview.steps++;

          if (result.done)
          {
              TraceLog(
                  LOG_INFO,
                  "Preview %d reached goal in %d steps",
                  preview.checkpoint,
                  preview.steps
              );

              preview.active = false;
          }
          else if (
              preview.steps >= PREVIEW_MAX_STEPS
          )
          {
              TraceLog(
                  LOG_INFO,
                  "Preview %d stopped after %d steps",
                  preview.checkpoint,
                  preview.steps
              );

              preview.active = false;
          }
      }
  }

        const char *statusText;
        Color statusColor;

        if (preview.active)
        {
            statusText = TextFormat(
                "Preview: %d (eps=0)",
                preview.checkpoint
            );

            statusColor = PURPLE;
        }
        else if (training)
        {
            statusText = "Training: RUNNING";
            statusColor = RED;
        }
        else
        {
            statusText = "Training: PAUSED";
            statusColor = DARKGRAY;
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawText(
            "Maze RL",
            20,
            20,
            30,
            BLACK
        );

        DrawMaze();

        if (showValues)
        {
            DrawValueHeatmap();
        }

        if (showPolicy)
        {
            DrawPolicy();
        }

        DrawAgent(agentPosition);

        // Display current state
        int currentState =
            PositionToState(agentPosition);

        DrawText(
            TextFormat(
                "Current State: %d",
                currentState
            ),
            200,
            10,
            18,
            BLACK
        );

        DrawText(
            TextFormat(
                "Epsilon: %.3f",
                epsilon
            ),
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
            statusText,
            430,
            35,
            18,
            statusColor
        );

        EndDrawing();
    }

    CloseWindow();

    return 0;
}