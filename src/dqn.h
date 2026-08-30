#ifndef DQN_H
#define DQN_H

#include <stdbool.h>
#include "learner.h"

typedef struct {
    float gamma;
    float learningRate;
    int replayCapacity;
    int replayWarmup;
    int batchSize;
    int targetSyncUpdates;
    float gradientClip;
} DqnConfig;

DqnConfig DefaultDqnConfig(void);
bool CreateDqnLearner(Learner *learner, DqnConfig config, Rng *rng);
bool DqnRunSelfTests(void);

#endif
