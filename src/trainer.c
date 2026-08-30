#include "trainer.h"
#include "agent.h"
#include "environment.h"
#include <string.h>

#define EPSILON_MIN 0.05f
#define EPSILON_DECAY 0.995f

void InitializeTrainingStats(TrainingStats *stats)
{
    memset(stats, 0, sizeof(*stats));
    stats->epsilon = 1.0f;
}

static EpisodeResult RunEpisode(Learner *learner, int episodeNumber, float epsilon, bool learn, Rng *rng)
{
    EpisodeResult result = { .episode = episodeNumber };
    int state = PositionToState(startPosition);
    for (int step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
        Action action = LearnerSelectAction(learner, state, epsilon, rng);
        StepResult outcome = EnvironmentStep(state, action);
        Transition transition = {state, action, outcome.reward, outcome.nextState, outcome.done};
        if (learn) LearnerObserve(learner, transition, rng);
        result.totalReward += outcome.reward;
        result.steps = step + 1;
        state = outcome.nextState;
        if (outcome.done) { result.reachedGoal = true; break; }
    }
    return result;
}

EpisodeResult TrainEpisode(Learner *learner, int episodeNumber, float epsilon, Rng *rng)
{
    return RunEpisode(learner, episodeNumber, epsilon, true, rng);
}

EpisodeResult EvaluateGreedy(Learner *learner, Rng *rng)
{
    return RunEpisode(learner, 0, 0.0f, false, rng);
}

void TrainAndRecordEpisode(Learner *learner, TrainingStats *stats, Rng *rng)
{
    stats->lastEpisode = TrainEpisode(learner, stats->episodes + 1, stats->epsilon, rng);
    int index = stats->episodes % METRIC_WINDOW;
    stats->successes[index] = stats->lastEpisode.reachedGoal;
    stats->steps[index] = stats->lastEpisode.steps;
    if (stats->recentCount < METRIC_WINDOW) stats->recentCount++;
    stats->episodes++;
    stats->epsilon *= EPSILON_DECAY;
    if (stats->epsilon < EPSILON_MIN) stats->epsilon = EPSILON_MIN;

    int successCount = 0;
    int totalSteps = 0;
    for (int metric = 0; metric < stats->recentCount; metric++) {
        if (stats->successes[metric]) { successCount++; totalSteps += stats->steps[metric]; }
    }
    stats->successRate = stats->recentCount > 0 ? 100.0f * successCount / stats->recentCount : 0.0f;
    stats->averageSuccessfulSteps = successCount > 0 ? (float)totalSteps / successCount : 0.0f;
}
