#include "learner.h"

void DestroyLearner(Learner *learner)
{
    if (learner != NULL && learner->state != NULL) {
        learner->ops.destroy(learner->state);
        learner->state = NULL;
    }
}

void ResetLearner(Learner *learner, Rng *rng)
{
    learner->ops.reset(learner->state, rng);
}

Action LearnerSelectAction(Learner *learner, int state, float epsilon, Rng *rng)
{
    return learner->ops.selectAction(learner->state, state, epsilon, rng);
}

void LearnerObserve(Learner *learner, Transition transition, Rng *rng)
{
    learner->ops.observe(learner->state, transition, rng);
}

void LearnerGetQValues(const Learner *learner, int state, float values[ACTION_COUNT])
{
    learner->ops.getQValues(learner->state, state, values);
}

float LearnerMaximumQ(const Learner *learner, int state)
{
    float values[ACTION_COUNT];
    LearnerGetQValues(learner, state, values);
    float maximum = values[0];
    for (int action = 1; action < ACTION_COUNT; action++) {
        if (values[action] > maximum) maximum = values[action];
    }
    return maximum;
}

size_t LearnerMemoryBytes(const Learner *learner)
{
    return learner->ops.memoryBytes(learner->state);
}

size_t LearnerParameterCount(const Learner *learner)
{
    return learner->ops.parameterCount(learner->state);
}
