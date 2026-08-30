#include "agent.h"
#include "dqn.h"
#include "environment.h"
#include "tabular.h"
#include "trainer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return false; } \
} while (0)

static bool TestEnvironment(void)
{
    int start = PositionToState(startPosition);
    StepResult right = EnvironmentStep(start, ACTION_RIGHT);
    CHECK(right.nextState == 12 && right.reward == -1.0f && !right.done,
          "moving right from start should enter state 12");
    StepResult wall = EnvironmentStep(start, ACTION_UP);
    CHECK(wall.nextState == start && wall.reward == -5.0f && !wall.done,
          "a wall collision should keep the state and cost -5");
    StepResult goal = EnvironmentStep(87, ACTION_RIGHT);
    CHECK(goal.nextState == 88 && goal.reward == 100.0f && goal.done,
          "entering the goal should terminate with +100");
    return true;
}

static bool TrainToOptimal(AgentKind kind, uint64_t seed)
{
    Rng rng;
    RngSeed(&rng, seed ^ ((uint64_t)kind << 48));
    Learner learner = {0};
    bool created = kind == AGENT_DQN ?
        CreateDqnLearner(&learner, DefaultDqnConfig(), &rng) :
        CreateTabularLearner(&learner, 0.1f, 0.95f);
    CHECK(created, "learner allocation should succeed");
    TrainingStats stats;
    InitializeTrainingStats(&stats);
    for (int episode = 0; episode < 1000; episode++)
        TrainAndRecordEpisode(&learner, &stats, &rng);
    Rng evaluationRng;
    RngSeed(&evaluationRng, seed ^ UINT64_C(0xa5a5a5a5));
    EpisodeResult evaluation = EvaluateGreedy(&learner, &evaluationRng);
    CHECK(evaluation.reachedGoal, "trained greedy policy should reach the goal");
    CHECK(evaluation.steps == 14, "trained greedy policy should use the optimal 14 steps");
    CHECK(LearnerParameterCount(&learner) == (kind == AGENT_DQN ? 6724u : 400u),
          "parameter count should match the selected representation");
    DestroyLearner(&learner);
    return true;
}

static bool TestTabularDeterminism(void)
{
    Learner first = {0};
    Learner second = {0};
    CHECK(CreateTabularLearner(&first, 0.1f, 0.95f), "first table allocation");
    CHECK(CreateTabularLearner(&second, 0.1f, 0.95f), "second table allocation");
    Rng firstRng;
    Rng secondRng;
    RngSeed(&firstRng, 42);
    RngSeed(&secondRng, 42);
    for (int episode = 1; episode <= 300; episode++) {
        float epsilon = fmaxf(0.05f, powf(0.995f, (float)(episode - 1)));
        TrainEpisode(&first, episode, epsilon, &firstRng);
        TrainEpisode(&second, episode, epsilon, &secondRng);
    }
    for (int state = 0; state < STATE_COUNT; state++) {
        float a[ACTION_COUNT];
        float b[ACTION_COUNT];
        LearnerGetQValues(&first, state, a);
        LearnerGetQValues(&second, state, b);
        for (int action = 0; action < ACTION_COUNT; action++)
            CHECK(a[action] == b[action], "same seed should produce identical Q-tables");
    }
    DestroyLearner(&first);
    DestroyLearner(&second);
    return true;
}

int main(void)
{
    InitializeAgent();
    if (!TestEnvironment()) return EXIT_FAILURE;
    if (!DqnRunSelfTests()) { fprintf(stderr, "FAIL: DQN internal self-tests\n"); return EXIT_FAILURE; }
    if (!TestTabularDeterminism()) return EXIT_FAILURE;
    if (!TrainToOptimal(AGENT_TABULAR, 7)) return EXIT_FAILURE;
    if (!TrainToOptimal(AGENT_DQN, 7)) return EXIT_FAILURE;
    printf("All RL tests passed.\n");
    return EXIT_SUCCESS;
}
