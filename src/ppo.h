#ifndef PPO_H
#define PPO_H

#include <stdbool.h>
#include <stdint.h>

/* Stage 1: PPO on the single fixed maze, for parity against the tabular and
   DQN learners already benchmarked on it. Self-contained (its own networks,
   rollout buffer and training loop) rather than implemented against
   learner.h, whose epsilon-greedy/Q-value interface does not fit an
   on-policy stochastic-policy method. See ppo_design.md. */
int RunPpoExperiment(int argc, char **argv);
bool PpoRunSelfTests(void);

/* A trained actor, saved by `--ppo --generalize --save-policy FILE` and
   replayed by the 3D view. Opaque so the network layout stays private, and
   so greedy action selection goes through the *same* forward pass training
   used -- a second copy could drift and then the rendered trajectory would
   no longer be the policy's actual behavior. */
typedef struct PpoPolicy PpoPolicy;

PpoPolicy *PpoPolicyLoad(const char *path);
void PpoPolicyDestroy(PpoPolicy *policy);
int PpoPolicyInputSize(const PpoPolicy *policy);

/* Argmax over the policy's logits for a layout observation. */
int PpoPolicyGreedyAction(const PpoPolicy *policy, const float *observation);

/* ---------- steppable trainer, for the live dashboard ----------

   The headless experiment drivers own a closed training loop. The 3D view
   needs to interleave training with rendering, so the same loop body is also
   exposed one rollout at a time. Both go through the identical collect ->
   GAE -> epochs sequence; this is a different caller, not a second
   implementation. */

typedef struct PpoTrainer PpoTrainer;

typedef struct {
    int environmentSteps;
    int updates;
    int episodes;          /* episodes completed during the last rollout */
    int goals;             /* how many of those reached the goal */
    float trainReturn;     /* mean undiscounted episode return, last rollout */
    float entropy;
    float valueLoss;
    float clipFraction;
    /* Schulman's k3 estimator: mean(exp(logr) - 1 - logr). Non-negative and
       far lower variance than mean(-logr), which can come out negative and
       look broken. */
    float approxKl;
} PpoTrainerMetrics;

PpoTrainer *PpoTrainerCreate(uint64_t seed, bool randomGoals, int minSeparation);
void PpoTrainerDestroy(PpoTrainer *trainer);

/* One rollout plus its epochs of minibatch updates. */
void PpoTrainerStep(PpoTrainer *trainer);
PpoTrainerMetrics PpoTrainerLastMetrics(const PpoTrainer *trainer);

/* Greedy rollout of the current policy on one suite maze, for display.
   Writes at most `capacity` states and returns how many. */
int PpoTrainerGreedyRoute(
    const PpoTrainer *trainer,
    int mazeIndex,
    int *route,
    int capacity,
    bool *reachedGoal);

typedef struct {
    float meanReturn;
    int solved;
    /* Failures that ended oscillating between two cells. Deterministic argmax
       can livelock even when the policy can see the goal: at A the best action
       leads to B and at B it leads back to A. Any exploration escapes it, so
       this measures a property of greedy evaluation, not of the policy being
       lost -- and it is the dominant failure mode here, so it is worth
       counting rather than lumping in with "did not solve". */
    int livelocked;
    int total;
} PpoEvalSummary;

/* Greedy rollouts over a contiguous run of suite mazes. Deliberately separate
   from training return: "is the learned policy improving", not "how is data
   collection going". */
PpoEvalSummary PpoTrainerEvaluate(const PpoTrainer *trainer, int firstMaze, int count);

#endif
