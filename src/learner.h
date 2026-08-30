#ifndef LEARNER_H
#define LEARNER_H

#include <stddef.h>
#include "rl.h"
#include "rng.h"

typedef struct Learner Learner;

typedef struct {
    void (*destroy)(void *state);
    void (*reset)(void *state, Rng *rng);
    Action (*selectAction)(void *state, int mazeState, float epsilon, Rng *rng);
    void (*observe)(void *state, Transition transition, Rng *rng);
    void (*getQValues)(const void *state, int mazeState, float values[ACTION_COUNT]);
    size_t (*memoryBytes)(const void *state);
    size_t (*parameterCount)(const void *state);
} LearnerOps;

struct Learner {
    AgentKind kind;
    void *state;
    LearnerOps ops;
};

void DestroyLearner(Learner *learner);
void ResetLearner(Learner *learner, Rng *rng);
Action LearnerSelectAction(Learner *learner, int state, float epsilon, Rng *rng);
void LearnerObserve(Learner *learner, Transition transition, Rng *rng);
void LearnerGetQValues(const Learner *learner, int state, float values[ACTION_COUNT]);
float LearnerMaximumQ(const Learner *learner, int state);
size_t LearnerMemoryBytes(const Learner *learner);
size_t LearnerParameterCount(const Learner *learner);

#endif
