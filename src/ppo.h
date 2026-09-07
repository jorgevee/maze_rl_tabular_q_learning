#ifndef PPO_H
#define PPO_H

#include <stdbool.h>

/* Stage 1: PPO on the single fixed maze, for parity against the tabular and
   DQN learners already benchmarked on it. Self-contained (its own networks,
   rollout buffer and training loop) rather than implemented against
   learner.h, whose epsilon-greedy/Q-value interface does not fit an
   on-policy stochastic-policy method. See ppo_design.md. */
int RunPpoExperiment(int argc, char **argv);
bool PpoRunSelfTests(void);

#endif
