#ifndef TRAINER_H
#define TRAINER_H

#include "learner.h"

#define METRIC_WINDOW 100

typedef struct {
    int episodes;
    float epsilon;
    bool successes[METRIC_WINDOW];
    int steps[METRIC_WINDOW];
    int recentCount;
    float successRate;
    float averageSuccessfulSteps;
    EpisodeResult lastEpisode;
} TrainingStats;

void InitializeTrainingStats(TrainingStats *stats);
EpisodeResult TrainEpisode(Learner *learner, int episodeNumber, float epsilon, Rng *rng);
EpisodeResult EvaluateGreedy(Learner *learner, Rng *rng);
void TrainAndRecordEpisode(Learner *learner, TrainingStats *stats, Rng *rng);

#endif
