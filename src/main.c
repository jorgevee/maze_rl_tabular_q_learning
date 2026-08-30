#include "raylib.h"
#include "maze.h"
#include "agent.h"
#include "benchmark.h"
#include "dqn.h"
#include "environment.h"
#include "tabular.h"
#include "trainer.h"
#include <stdio.h>

#define TRAINING_EPISODES 5000
#define PREVIEW_MAX_STEPS 60

typedef struct {
    bool active;
    int state;
    int steps;
    float timer;
} PolicyPreview;

static bool CreateLearners(Learner learners[AGENT_KIND_COUNT], Rng rngs[AGENT_KIND_COUNT])
{
    RngSeed(&rngs[AGENT_TABULAR], 1);
    RngSeed(&rngs[AGENT_DQN], 2);
    if (!CreateTabularLearner(&learners[AGENT_TABULAR], 0.1f, 0.95f)) return false;
    if (!CreateDqnLearner(&learners[AGENT_DQN], DefaultDqnConfig(), &rngs[AGENT_DQN])) {
        DestroyLearner(&learners[AGENT_TABULAR]);
        return false;
    }
    return true;
}

static void StartPreview(PolicyPreview *preview)
{
    *preview = (PolicyPreview){
        .active = true,
        .state = PositionToState(startPosition)
    };
    ResetAgent();
}

int main(int argc, char **argv)
{
    InitializeAgent();
    BenchmarkOptions benchmark;
    if (argc > 1) {
        if (!ParseBenchmarkOptions(argc, argv, &benchmark)) {
            fprintf(stderr, "Usage: %s --benchmark [--agent both|tabular|dqn] [--episodes N] [--seeds N] [--seed N] [--csv FILE]\n", argv[0]);
            return 2;
        }
        return RunBenchmark(&benchmark);
    }

    Learner learners[AGENT_KIND_COUNT] = {0};
    Rng rngs[AGENT_KIND_COUNT];
    if (!CreateLearners(learners, rngs)) {
        fprintf(stderr, "Could not allocate learners.\n");
        return 1;
    }
    TrainingStats stats[AGENT_KIND_COUNT];
    InitializeTrainingStats(&stats[AGENT_TABULAR]);
    InitializeTrainingStats(&stats[AGENT_DQN]);

    InitWindow(800, 800, "Tabular Q-Learning vs DQN");
    SetTargetFPS(60);
    AgentKind selected = AGENT_TABULAR;
    bool training = false;
    bool humanMode = true;
    bool showPolicy = false;
    bool showValues = false;
    int episodesPerFrame = 5;
    PolicyPreview preview = {0};

    while (!WindowShouldClose()) {
        Learner *learner = &learners[selected];
        TrainingStats *currentStats = &stats[selected];
        Rng *rng = &rngs[selected];

        if (IsKeyPressed(KEY_A)) {
            training = false;
            preview.active = false;
            selected = selected == AGENT_TABULAR ? AGENT_DQN : AGENT_TABULAR;
            ResetAgent();
            learner = &learners[selected];
            currentStats = &stats[selected];
            rng = &rngs[selected];
        }
        if (IsKeyPressed(KEY_H)) { humanMode = true; training = false; preview.active = false; }
        if (IsKeyPressed(KEY_T) && currentStats->episodes < TRAINING_EPISODES) {
            training = !training; humanMode = false; preview.active = false;
        }
        if (IsKeyPressed(KEY_D)) { training = false; humanMode = false; StartPreview(&preview); }
        if (IsKeyPressed(KEY_R)) ResetAgent();
        if (IsKeyPressed(KEY_C)) {
            training = false; preview.active = false;
            ResetLearner(learner, rng);
            InitializeTrainingStats(currentStats);
            ResetAgent();
        }
        if (IsKeyPressed(KEY_P)) showPolicy = !showPolicy;
        if (IsKeyPressed(KEY_V)) showValues = !showValues;
        if ((IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) && episodesPerFrame < 1000)
            episodesPerFrame *= 2;
        if ((IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) && episodesPerFrame > 1)
            episodesPerFrame = episodesPerFrame > 2 ? episodesPerFrame / 2 : 1;

        if (humanMode && !training && !preview.active) {
            if (IsKeyPressed(KEY_UP)) TryMove(&agentPosition, ACTION_UP);
            else if (IsKeyPressed(KEY_RIGHT)) TryMove(&agentPosition, ACTION_RIGHT);
            else if (IsKeyPressed(KEY_DOWN)) TryMove(&agentPosition, ACTION_DOWN);
            else if (IsKeyPressed(KEY_LEFT)) TryMove(&agentPosition, ACTION_LEFT);
        }

        if (training) {
            for (int episode = 0; episode < episodesPerFrame && currentStats->episodes < TRAINING_EPISODES; episode++)
                TrainAndRecordEpisode(learner, currentStats, rng);
            if (currentStats->episodes >= TRAINING_EPISODES) training = false;
        }

        if (preview.active) {
            preview.timer += GetFrameTime();
            if (preview.timer >= 0.10f) {
                preview.timer = 0.0f;
                Action action = LearnerSelectAction(learner, preview.state, 0.0f, rng);
                StepResult result = EnvironmentStep(preview.state, action);
                preview.state = result.nextState;
                agentPosition = StateToPosition(preview.state);
                preview.steps++;
                if (result.done || preview.steps >= PREVIEW_MAX_STEPS) preview.active = false;
            }
        }

        const char *mode = preview.active ? "GREEDY DEMO (eps=0)" :
            training ? "TRAINING" : humanMode ? "HUMAN" : "PAUSED";
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText(TextFormat("Agent: %s | Mode: %s", AgentKindName(selected), mode), 20, 8, 18, BLACK);
        DrawText(TextFormat("Episodes: %d/%d  eps: %.3f  speed: %d/frame",
            currentStats->episodes, TRAINING_EPISODES, currentStats->epsilon, episodesPerFrame), 20, 30, 17, DARKGRAY);
        DrawText(TextFormat("Last 100: %.0f%% success  %.2f avg successful steps",
            currentStats->successRate, currentStats->averageSuccessfulSteps), 400, 52, 15, DARKGRAY);
        DrawMaze();
        if (showValues) DrawValueHeatmap(learner);
        if (showPolicy) DrawPolicy(learner);
        DrawAgent(agentPosition);
        DrawText("A agent  H human  T train  D demo  C clear  R reset  P policy  V values  +/- speed",
            18, 778, 13, BLACK);
        EndDrawing();
    }

    DestroyLearner(&learners[AGENT_TABULAR]);
    DestroyLearner(&learners[AGENT_DQN]);
    CloseWindow();
    return 0;
}
