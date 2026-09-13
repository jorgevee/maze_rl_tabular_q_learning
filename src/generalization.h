#ifndef GENERALIZATION_H
#define GENERALIZATION_H

#include <stdbool.h>
#include "mazesuite.h"

/* The DQN held-out generalization experiments. The environment they run on
   (maze suite, observation encoding, step function) lives in mazesuite.h,
   shared with the other algorithms evaluated on the same suite. */
int RunGeneralizationExperiment(int argc, char **argv);
int RunGeneralizationVideo(int argc, char **argv);
bool GeneralizationRunSelfTests(void);

#endif
