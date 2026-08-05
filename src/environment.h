// This header exposes the environment’s public interface.
#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdbool.h>
#include "agent.h"

#define MAX_STEPS_PER_EPISODE 200

typedef struct {
    int nextState;
    float reward;
    bool done;
} StepResult;

StepResult EnvironmentStep(int state, Action action);

#endif
