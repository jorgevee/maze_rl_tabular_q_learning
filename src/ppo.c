#include "ppo.h"
#include "agent.h"
#include "dqn.h"
#include "environment.h"
#include "learner.h"
#include "maze.h"
#include "rl.h"
#include "rng.h"
#include "trainer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PPO_HIDDEN 64
#define PPO_ROLLOUT_CAPACITY 4096
#define PPO_ADAM_BETA1 0.9f
#define PPO_ADAM_BETA2 0.999f
#define PPO_ADAM_EPSILON 1.0e-8f

/* Actor: one-hot state -> hidden ReLU -> ACTION_COUNT logits. Deliberately
   the same shape (and therefore the same 6,724 parameters) as the
   single-maze DQN it is compared against, so a DQN-vs-PPO result is not
   also a network-size comparison. */
typedef struct {
    float w1[PPO_HIDDEN][STATE_COUNT];
    float b1[PPO_HIDDEN];
    float w2[ACTION_COUNT][PPO_HIDDEN];
    float b2[ACTION_COUNT];
} ActorNetwork;

/* Critic: same trunk shape, one scalar state-value output (6,529 parameters).
   Kept as a separate network rather than a shared trunk with two heads: two
   independent forward/backward passes are individually verifiable by
   finite differences, which matters more here than the parameter saving. */
typedef struct {
    float w1[PPO_HIDDEN][STATE_COUNT];
    float b1[PPO_HIDDEN];
    float w2[PPO_HIDDEN];
    float b2;
} CriticNetwork;

typedef struct {
    int state;
    Action action;
    float logProbOld;   /* log pi(a|s) under the policy that collected this step */
    float reward;
    float value;        /* V(s) at collection time */
    float nextValue;    /* V(s') at collection time; exactly 0 when the goal was reached */
    bool episodeEnd;    /* goal reached OR truncated: stop chaining advantage across it */
    float advantage;
    float target;       /* advantage + value, recorded before advantage normalization */
} PpoStep;

typedef struct {
    float gamma;
    float lambda;
    float clipEpsilon;
    float learningRate;
    float entropyCoefficient;
    float gradientClip;
    int rolloutSteps;
    int epochs;
    int minibatchSize;
} PpoConfig;

typedef struct {
    ActorNetwork actor;
    ActorNetwork actorFirstMoment;
    ActorNetwork actorSecondMoment;
    CriticNetwork critic;
    CriticNetwork criticFirstMoment;
    CriticNetwork criticSecondMoment;
    int actorUpdates;
    int criticUpdates;
    PpoConfig config;
    PpoStep steps[PPO_ROLLOUT_CAPACITY];
    int stepCount;
    int environmentSteps;
    /* Rollout collection continues across batches instead of restarting at
       the maze start every time, so a batch boundary does not silently
       truncate an episode. */
    int currentState;
    int currentEpisodeSteps;
    float currentEpisodeReturn;
    /* Per-rollout episode statistics. */
    float completedReturnSum;
    int completedEpisodes;
    int completedGoals;
} PpoAgent;

typedef struct {
    float entropy;
    float valueLoss;
    float clipFraction;
} PpoBatchStats;

static PpoConfig DefaultPpoConfig(void)
{
    return (PpoConfig){
        .gamma = 0.95f,              /* matches the discount used everywhere else here */
        .lambda = 0.95f,             /* GAE smoothing, standard default */
        .clipEpsilon = 0.2f,         /* standard PPO clip range */
        .learningRate = 3.0e-4f,     /* standard PPO default (DQN here uses 1e-3) */
        .entropyCoefficient = 0.01f,
        .gradientClip = 0.5f,
        .rolloutSteps = 2048,
        .epochs = 4,
        .minibatchSize = 64
    };
}

/* ---------- networks ---------- */

static void InitializeActor(ActorNetwork *actor, Rng *rng)
{
    memset(actor, 0, sizeof(*actor));
    const float firstScale = sqrtf(2.0f / STATE_COUNT);
    const float secondScale = sqrtf(2.0f / PPO_HIDDEN);
    for (int hidden = 0; hidden < PPO_HIDDEN; hidden++) {
        for (int input = 0; input < STATE_COUNT; input++)
            actor->w1[hidden][input] = RngNormal(rng) * firstScale;
    }
    for (int action = 0; action < ACTION_COUNT; action++) {
        for (int hidden = 0; hidden < PPO_HIDDEN; hidden++)
            actor->w2[action][hidden] = RngNormal(rng) * secondScale;
    }
}

static void InitializeCritic(CriticNetwork *critic, Rng *rng)
{
    memset(critic, 0, sizeof(*critic));
    const float firstScale = sqrtf(2.0f / STATE_COUNT);
    const float secondScale = sqrtf(2.0f / PPO_HIDDEN);
    for (int hidden = 0; hidden < PPO_HIDDEN; hidden++) {
        for (int input = 0; input < STATE_COUNT; input++)
            critic->w1[hidden][input] = RngNormal(rng) * firstScale;
        critic->w2[hidden] = RngNormal(rng) * secondScale;
    }
}

/* The input is one-hot over states, so the first layer is a column lookup
   rather than a full matrix multiply (same trick dqn.c uses). */
static void ActorForward(
    const ActorNetwork *actor,
    int state,
    float hiddenValues[PPO_HIDDEN],
    float logits[ACTION_COUNT])
{
    for (int hidden = 0; hidden < PPO_HIDDEN; hidden++) {
        float value = actor->w1[hidden][state] + actor->b1[hidden];
        hiddenValues[hidden] = value > 0.0f ? value : 0.0f;
    }
    for (int action = 0; action < ACTION_COUNT; action++) {
        float value = actor->b2[action];
        for (int hidden = 0; hidden < PPO_HIDDEN; hidden++)
            value += actor->w2[action][hidden] * hiddenValues[hidden];
        logits[action] = value;
    }
}

static float CriticForward(
    const CriticNetwork *critic,
    int state,
    float hiddenValues[PPO_HIDDEN])
{
    float value = critic->b2;
    for (int hidden = 0; hidden < PPO_HIDDEN; hidden++) {
        float activation = critic->w1[hidden][state] + critic->b1[hidden];
        hiddenValues[hidden] = activation > 0.0f ? activation : 0.0f;
        value += critic->w2[hidden] * hiddenValues[hidden];
    }
    return value;
}

/* ---------- categorical policy ---------- */

static void PolicyDistribution(
    const float logits[ACTION_COUNT],
    float probabilities[ACTION_COUNT],
    float logProbabilities[ACTION_COUNT])
{
    float maximum = logits[0];
    for (int action = 1; action < ACTION_COUNT; action++)
        if (logits[action] > maximum) maximum = logits[action];
    float sum = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++) {
        probabilities[action] = expf(logits[action] - maximum);
        sum += probabilities[action];
    }
    float logSum = logf(sum);
    for (int action = 0; action < ACTION_COUNT; action++) {
        logProbabilities[action] = (logits[action] - maximum) - logSum;
        probabilities[action] /= sum;
    }
}

static float PolicyEntropy(
    const float probabilities[ACTION_COUNT],
    const float logProbabilities[ACTION_COUNT])
{
    float entropy = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++)
        entropy -= probabilities[action] * logProbabilities[action];
    return entropy;
}

/* Exploration in PPO comes from sampling this distribution, not from an
   epsilon schedule bolted on outside the policy. */
static Action SampleAction(const float probabilities[ACTION_COUNT], Rng *rng)
{
    float sample = RngFloat(rng);
    float cumulative = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++) {
        cumulative += probabilities[action];
        if (sample < cumulative) return (Action)action;
    }
    return (Action)(ACTION_COUNT - 1);
}

/* ---------- losses ---------- */

static float HuberDerivative(float error)
{
    if (error > 1.0f) return 1.0f;
    if (error < -1.0f) return -1.0f;
    return error;
}

static float HuberLoss(float error)
{
    float absolute = fabsf(error);
    return absolute <= 1.0f ? 0.5f * error * error : absolute - 0.5f;
}

static float ClipRatio(float ratio, float clipEpsilon)
{
    if (ratio < 1.0f - clipEpsilon) return 1.0f - clipEpsilon;
    if (ratio > 1.0f + clipEpsilon) return 1.0f + clipEpsilon;
    return ratio;
}

/* Scalar actor loss for one sample, used by the finite-difference self-test
   and mirrored exactly by AccumulateActorGradient. */
static float ActorLossFor(
    const ActorNetwork *actor,
    const PpoStep *step,
    float advantage,
    const PpoConfig *config)
{
    float hidden[PPO_HIDDEN];
    float logits[ACTION_COUNT];
    ActorForward(actor, step->state, hidden, logits);
    float probabilities[ACTION_COUNT];
    float logProbabilities[ACTION_COUNT];
    PolicyDistribution(logits, probabilities, logProbabilities);

    float ratio = expf(logProbabilities[step->action] - step->logProbOld);
    float clipped = ClipRatio(ratio, config->clipEpsilon);
    float surrogate1 = ratio * advantage;
    float surrogate2 = clipped * advantage;
    float objective = surrogate1 <= surrogate2 ? surrogate1 : surrogate2;
    float entropy = PolicyEntropy(probabilities, logProbabilities);
    return -(objective + config->entropyCoefficient * entropy);
}

static void AccumulateActorGradient(
    const ActorNetwork *actor,
    const PpoStep *step,
    float advantage,
    const PpoConfig *config,
    float scale,
    ActorNetwork *gradient,
    float *entropyOut,
    bool *clippedOut)
{
    float hidden[PPO_HIDDEN];
    float logits[ACTION_COUNT];
    ActorForward(actor, step->state, hidden, logits);
    float probabilities[ACTION_COUNT];
    float logProbabilities[ACTION_COUNT];
    PolicyDistribution(logits, probabilities, logProbabilities);

    float ratio = expf(logProbabilities[step->action] - step->logProbOld);
    float clipped = ClipRatio(ratio, config->clipEpsilon);
    float surrogate1 = ratio * advantage;
    float surrogate2 = clipped * advantage;
    /* PPO takes the pessimistic (minimum) branch. When the clipped branch
       wins, the objective is constant in the policy parameters over that
       region and contributes no gradient at all -- that is the entire
       mechanism keeping repeated epochs on one batch from running away. */
    bool useUnclipped = surrogate1 <= surrogate2;
    float objectiveGradient = useUnclipped ? ratio * advantage : 0.0f;
    float entropy = PolicyEntropy(probabilities, logProbabilities);

    /* loss = -(objective + c * entropy)
         d(logProb[a*])/d(logit[j]) = (j == a*) - p[j]
         d(entropy)/d(logit[j])     = -p[j] * (logProb[j] + entropy)          */
    float logitGradient[ACTION_COUNT];
    for (int action = 0; action < ACTION_COUNT; action++) {
        float logProbGradient = (action == (int)step->action ? 1.0f : 0.0f) -
            probabilities[action];
        float entropyGradient = -probabilities[action] *
            (logProbabilities[action] + entropy);
        logitGradient[action] = -(objectiveGradient * logProbGradient +
            config->entropyCoefficient * entropyGradient) * scale;
    }

    for (int action = 0; action < ACTION_COUNT; action++) {
        gradient->b2[action] += logitGradient[action];
        for (int hiddenIndex = 0; hiddenIndex < PPO_HIDDEN; hiddenIndex++)
            gradient->w2[action][hiddenIndex] += logitGradient[action] * hidden[hiddenIndex];
    }
    for (int hiddenIndex = 0; hiddenIndex < PPO_HIDDEN; hiddenIndex++) {
        if (hidden[hiddenIndex] <= 0.0f) continue;
        float hiddenGradient = 0.0f;
        for (int action = 0; action < ACTION_COUNT; action++)
            hiddenGradient += logitGradient[action] * actor->w2[action][hiddenIndex];
        gradient->b1[hiddenIndex] += hiddenGradient;
        gradient->w1[hiddenIndex][step->state] += hiddenGradient;
    }

    if (entropyOut) *entropyOut = entropy;
    if (clippedOut) *clippedOut = !useUnclipped;
}

static float CriticLossFor(const CriticNetwork *critic, int state, float target)
{
    float hidden[PPO_HIDDEN];
    float value = CriticForward(critic, state, hidden);
    return HuberLoss(value - target);
}

static void AccumulateCriticGradient(
    const CriticNetwork *critic,
    int state,
    float target,
    float scale,
    CriticNetwork *gradient,
    float *lossOut)
{
    float hidden[PPO_HIDDEN];
    float value = CriticForward(critic, state, hidden);
    float error = value - target;
    float outputGradient = HuberDerivative(error) * scale;

    gradient->b2 += outputGradient;
    for (int hiddenIndex = 0; hiddenIndex < PPO_HIDDEN; hiddenIndex++) {
        gradient->w2[hiddenIndex] += outputGradient * hidden[hiddenIndex];
        if (hidden[hiddenIndex] > 0.0f) {
            float hiddenGradient = outputGradient * critic->w2[hiddenIndex];
            gradient->b1[hiddenIndex] += hiddenGradient;
            gradient->w1[hiddenIndex][state] += hiddenGradient;
        }
    }
    if (lossOut) *lossOut = HuberLoss(error);
}

/* ---------- optimizer ---------- */

/* Both networks are flat float blocks, so one Adam implementation serves
   both (same approach dqn.c takes with its single Network struct). */
static void AdamStep(
    float *parameters,
    float *firstMoment,
    float *secondMoment,
    float *gradient,
    size_t count,
    int updates,
    float learningRate,
    float gradientClip)
{
    double normSquared = 0.0;
    for (size_t index = 0; index < count; index++)
        normSquared += (double)gradient[index] * gradient[index];
    float norm = (float)sqrt(normSquared);
    if (norm > gradientClip && norm > 0.0f) {
        float scale = gradientClip / norm;
        for (size_t index = 0; index < count; index++) gradient[index] *= scale;
    }

    float correction1 = 1.0f - powf(PPO_ADAM_BETA1, (float)updates);
    float correction2 = 1.0f - powf(PPO_ADAM_BETA2, (float)updates);
    for (size_t index = 0; index < count; index++) {
        firstMoment[index] = PPO_ADAM_BETA1 * firstMoment[index] +
            (1.0f - PPO_ADAM_BETA1) * gradient[index];
        secondMoment[index] = PPO_ADAM_BETA2 * secondMoment[index] +
            (1.0f - PPO_ADAM_BETA2) * gradient[index] * gradient[index];
        float firstHat = firstMoment[index] / correction1;
        float secondHat = secondMoment[index] / correction2;
        parameters[index] -= learningRate * firstHat / (sqrtf(secondHat) + PPO_ADAM_EPSILON);
    }
}

/* ---------- rollout, advantages, training ---------- */

static void ResetRolloutPosition(PpoAgent *agent)
{
    agent->currentState = PositionToState(startPosition);
    agent->currentEpisodeSteps = 0;
    agent->currentEpisodeReturn = 0.0f;
}

static void CollectRollout(PpoAgent *agent, Rng *rng)
{
    agent->stepCount = 0;
    agent->completedReturnSum = 0.0f;
    agent->completedEpisodes = 0;
    agent->completedGoals = 0;

    for (int index = 0; index < agent->config.rolloutSteps; index++) {
        int state = agent->currentState;
        float actorHidden[PPO_HIDDEN];
        float logits[ACTION_COUNT];
        ActorForward(&agent->actor, state, actorHidden, logits);
        float probabilities[ACTION_COUNT];
        float logProbabilities[ACTION_COUNT];
        PolicyDistribution(logits, probabilities, logProbabilities);
        Action action = SampleAction(probabilities, rng);

        float criticHidden[PPO_HIDDEN];
        float value = CriticForward(&agent->critic, state, criticHidden);

        StepResult outcome = EnvironmentStep(state, action);
        agent->environmentSteps++;
        agent->currentEpisodeSteps++;
        agent->currentEpisodeReturn += outcome.reward;
        bool truncated = !outcome.done &&
            agent->currentEpisodeSteps >= MAX_STEPS_PER_EPISODE;

        PpoStep *step = &agent->steps[agent->stepCount++];
        step->state = state;
        step->action = action;
        step->logProbOld = logProbabilities[action];
        step->reward = outcome.reward;
        step->value = value;
        step->episodeEnd = outcome.done || truncated;
        if (outcome.done) {
            /* Terminal: nothing follows the goal, so no bootstrap value. */
            step->nextValue = 0.0f;
        } else {
            /* Includes truncation, which is a limit of the harness rather
               than of the MDP, so it still bootstraps from where we stopped. */
            float nextHidden[PPO_HIDDEN];
            step->nextValue = CriticForward(&agent->critic, outcome.nextState, nextHidden);
        }

        if (step->episodeEnd) {
            agent->completedReturnSum += agent->currentEpisodeReturn;
            agent->completedEpisodes++;
            if (outcome.done) agent->completedGoals++;
            ResetRolloutPosition(agent);
        } else {
            agent->currentState = outcome.nextState;
        }
    }
}

static void ComputeAdvantages(PpoStep *steps, int count, const PpoConfig *config)
{
    float nextAdvantage = 0.0f;
    for (int index = count - 1; index >= 0; index--) {
        /* Advantage does not chain across an episode boundary. */
        if (steps[index].episodeEnd) nextAdvantage = 0.0f;
        float delta = steps[index].reward +
            config->gamma * steps[index].nextValue - steps[index].value;
        steps[index].advantage = delta + config->gamma * config->lambda * nextAdvantage;
        nextAdvantage = steps[index].advantage;
        /* Recorded before normalization: this is the critic's target. */
        steps[index].target = steps[index].advantage + steps[index].value;
    }
}

static void NormalizeAdvantages(PpoStep *steps, int count)
{
    if (count < 2) return;
    double sum = 0.0;
    for (int index = 0; index < count; index++) sum += steps[index].advantage;
    float mean = (float)(sum / count);
    double varianceSum = 0.0;
    for (int index = 0; index < count; index++) {
        double difference = (double)steps[index].advantage - mean;
        varianceSum += difference * difference;
    }
    float deviation = (float)sqrt(varianceSum / count);
    for (int index = 0; index < count; index++)
        steps[index].advantage = (steps[index].advantage - mean) / (deviation + 1.0e-8f);
}

static void TrainOnRollout(PpoAgent *agent, Rng *rng, PpoBatchStats *stats)
{
    static int order[PPO_ROLLOUT_CAPACITY];
    for (int index = 0; index < agent->stepCount; index++) order[index] = index;

    double entropySum = 0.0;
    double valueLossSum = 0.0;
    long clippedCount = 0;
    long sampleCount = 0;

    for (int epoch = 0; epoch < agent->config.epochs; epoch++) {
        for (int index = agent->stepCount - 1; index > 0; index--) {
            int other = RngRange(rng, index + 1);
            int swap = order[index];
            order[index] = order[other];
            order[other] = swap;
        }

        for (int start = 0; start < agent->stepCount; start += agent->config.minibatchSize) {
            int end = start + agent->config.minibatchSize;
            if (end > agent->stepCount) end = agent->stepCount;
            int size = end - start;
            if (size <= 0) continue;
            float scale = 1.0f / (float)size;

            ActorNetwork actorGradient;
            CriticNetwork criticGradient;
            memset(&actorGradient, 0, sizeof(actorGradient));
            memset(&criticGradient, 0, sizeof(criticGradient));

            for (int index = start; index < end; index++) {
                PpoStep *step = &agent->steps[order[index]];
                float entropy = 0.0f;
                bool wasClipped = false;
                AccumulateActorGradient(&agent->actor, step, step->advantage,
                    &agent->config, scale, &actorGradient, &entropy, &wasClipped);
                float valueLoss = 0.0f;
                AccumulateCriticGradient(&agent->critic, step->state, step->target,
                    scale, &criticGradient, &valueLoss);
                entropySum += entropy;
                valueLossSum += valueLoss;
                if (wasClipped) clippedCount++;
                sampleCount++;
            }

            agent->actorUpdates++;
            AdamStep((float *)&agent->actor, (float *)&agent->actorFirstMoment,
                (float *)&agent->actorSecondMoment, (float *)&actorGradient,
                sizeof(ActorNetwork) / sizeof(float), agent->actorUpdates,
                agent->config.learningRate, agent->config.gradientClip);

            agent->criticUpdates++;
            AdamStep((float *)&agent->critic, (float *)&agent->criticFirstMoment,
                (float *)&agent->criticSecondMoment, (float *)&criticGradient,
                sizeof(CriticNetwork) / sizeof(float), agent->criticUpdates,
                agent->config.learningRate, agent->config.gradientClip);
        }
    }

    if (stats && sampleCount > 0) {
        stats->entropy = (float)(entropySum / (double)sampleCount);
        stats->valueLoss = (float)(valueLossSum / (double)sampleCount);
        stats->clipFraction = (float)((double)clippedCount / (double)sampleCount);
    }
}

/* Deterministic argmax rollout, matching how the other learners here are
   evaluated. Uses no RNG, so evaluation cannot perturb training. */
static EpisodeResult EvaluateGreedyPolicy(const PpoAgent *agent)
{
    EpisodeResult result = {0};
    int state = PositionToState(startPosition);
    for (int step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
        float hidden[PPO_HIDDEN];
        float logits[ACTION_COUNT];
        ActorForward(&agent->actor, state, hidden, logits);
        int best = 0;
        for (int action = 1; action < ACTION_COUNT; action++)
            if (logits[action] > logits[best]) best = action;
        StepResult outcome = EnvironmentStep(state, (Action)best);
        result.totalReward += outcome.reward;
        result.steps = step + 1;
        state = outcome.nextState;
        if (outcome.done) { result.reachedGoal = true; break; }
    }
    return result;
}

static void InitializeAgentNetworks(PpoAgent *agent, PpoConfig config, Rng *rng)
{
    memset(agent, 0, sizeof(*agent));
    agent->config = config;
    InitializeActor(&agent->actor, rng);
    InitializeCritic(&agent->critic, rng);
    ResetRolloutPosition(agent);
}

/* ---------- experiment driver ---------- */

typedef struct {
    int totalSteps;
    int seeds;
    uint64_t firstSeed;
    const char *csvPath;
    bool compareDqn;
} PpoOptions;

static PpoOptions DefaultPpoOptions(void)
{
    return (PpoOptions){200000, 5, 1, "ppo.csv", false};
}

static bool ParsePpoPositive(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > 100000000) return false;
    *value = (int)parsed;
    return true;
}

static bool ParsePpoOptions(int argc, char **argv, PpoOptions *options)
{
    *options = DefaultPpoOptions();
    for (int index = 2; index < argc; index++) {
        if (strcmp(argv[index], "--steps") == 0 && index + 1 < argc) {
            if (!ParsePpoPositive(argv[++index], &options->totalSteps)) return false;
        } else if (strcmp(argv[index], "--seeds") == 0 && index + 1 < argc) {
            if (!ParsePpoPositive(argv[++index], &options->seeds)) return false;
        } else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            int seed;
            if (!ParsePpoPositive(argv[++index], &seed)) return false;
            options->firstSeed = (uint64_t)seed;
        } else if (strcmp(argv[index], "--csv") == 0 && index + 1 < argc) {
            options->csvPath = argv[++index];
        } else if (strcmp(argv[index], "--compare-dqn") == 0) {
            options->compareDqn = true;
        } else return false;
    }
    return true;
}

static int RunPpoSeed(FILE *csv, const PpoOptions *options, uint64_t seed)
{
    Rng rng;
    RngSeed(&rng, seed ^ UINT64_C(0x5052504f));
    PpoAgent *agent = malloc(sizeof(PpoAgent));
    if (!agent) return 1;
    InitializeAgentNetworks(agent, DefaultPpoConfig(), &rng);

    clock_t start = clock();
    int firstSolvedStep = -1;
    EpisodeResult evaluation = {0};
    while (agent->environmentSteps < options->totalSteps) {
        CollectRollout(agent, &rng);
        ComputeAdvantages(agent->steps, agent->stepCount, &agent->config);
        NormalizeAdvantages(agent->steps, agent->stepCount);
        PpoBatchStats stats = {0};
        TrainOnRollout(agent, &rng, &stats);

        evaluation = EvaluateGreedyPolicy(agent);
        if (firstSolvedStep < 0 && evaluation.reachedGoal)
            firstSolvedStep = agent->environmentSteps;
        double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
        float meanReturn = agent->completedEpisodes > 0 ?
            agent->completedReturnSum / (float)agent->completedEpisodes : 0.0f;

        fprintf(csv, "ppo,%llu,%d,%d,%d,%d,%.1f,%.2f,%d,%d,%.4f,%.4f,%.4f,%.3f\n",
            (unsigned long long)seed,
            agent->environmentSteps,
            agent->actorUpdates,
            evaluation.reachedGoal ? 1 : 0,
            evaluation.steps,
            evaluation.totalReward,
            meanReturn,
            agent->completedEpisodes,
            agent->completedGoals,
            stats.entropy,
            stats.valueLoss,
            stats.clipFraction,
            elapsed);
    }

    double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("seed=%llu algorithm=ppo steps=%d updates=%d greedy=%s steps_to_goal=%d first_solved_at=%d time=%.0fms\n",
        (unsigned long long)seed,
        agent->environmentSteps,
        agent->actorUpdates,
        evaluation.reachedGoal ? "solved" : "failed",
        evaluation.reachedGoal ? evaluation.steps : -1,
        firstSolvedStep,
        elapsed);
    free(agent);
    return 0;
}

/* DQN baseline on the same maze, same environment-step budget, and the same
   evaluation cadence, so the two algorithms are measured on one axis.
   Reuses the existing verified training path (trainer.c + dqn.c) rather than
   reimplementing it, and does not touch benchmark.c, whose results are
   already published. Environment steps -- not episodes -- are the shared
   unit: an episode means different amounts of experience to each algorithm,
   and one PPO update consumes a whole rollout while one DQN update consumes
   a replay minibatch. */
static int RunDqnBaselineSeed(FILE *csv, const PpoOptions *options, uint64_t seed)
{
    Rng trainingRng;
    RngSeed(&trainingRng, seed ^ ((uint64_t)AGENT_DQN << 48));
    Learner learner = {0};
    if (!CreateDqnLearner(&learner, DefaultDqnConfig(), &trainingRng)) return 1;

    TrainingStats stats;
    InitializeTrainingStats(&stats);
    Rng evaluationRng;
    clock_t start = clock();

    const int checkpointInterval = DefaultPpoConfig().rolloutSteps;
    int environmentSteps = 0;
    int nextCheckpoint = checkpointInterval;
    int firstSolvedStep = -1;
    int episodesSinceCheckpoint = 0;
    int goalsSinceCheckpoint = 0;
    float returnSinceCheckpoint = 0.0f;
    EpisodeResult evaluation = {0};

    while (environmentSteps < options->totalSteps) {
        TrainAndRecordEpisode(&learner, &stats, &trainingRng);
        environmentSteps += stats.lastEpisode.steps;
        episodesSinceCheckpoint++;
        returnSinceCheckpoint += stats.lastEpisode.totalReward;
        if (stats.lastEpisode.reachedGoal) goalsSinceCheckpoint++;

        if (environmentSteps >= nextCheckpoint) {
            RngSeed(&evaluationRng, seed ^ (uint64_t)environmentSteps ^ UINT64_C(0xa5a5a5a5));
            evaluation = EvaluateGreedy(&learner, &evaluationRng);
            if (firstSolvedStep < 0 && evaluation.reachedGoal)
                firstSolvedStep = environmentSteps;
            double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
            float meanReturn = episodesSinceCheckpoint > 0 ?
                returnSinceCheckpoint / (float)episodesSinceCheckpoint : 0.0f;
            /* Entropy, value loss and clip fraction are PPO-specific and are
               written as 0 for DQN rows rather than left blank. */
            fprintf(csv, "dqn,%llu,%d,%d,%d,%d,%.1f,%.2f,%d,%d,%.4f,%.4f,%.4f,%.3f\n",
                (unsigned long long)seed,
                environmentSteps,
                0,
                evaluation.reachedGoal ? 1 : 0,
                evaluation.steps,
                evaluation.totalReward,
                meanReturn,
                episodesSinceCheckpoint,
                goalsSinceCheckpoint,
                0.0f, 0.0f, 0.0f,
                elapsed);
            while (nextCheckpoint <= environmentSteps) nextCheckpoint += checkpointInterval;
            episodesSinceCheckpoint = 0;
            goalsSinceCheckpoint = 0;
            returnSinceCheckpoint = 0.0f;
        }
    }

    double elapsed = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("seed=%llu algorithm=dqn steps=%d episodes=%d greedy=%s steps_to_goal=%d first_solved_at=%d time=%.0fms\n",
        (unsigned long long)seed,
        environmentSteps,
        stats.episodes,
        evaluation.reachedGoal ? "solved" : "failed",
        evaluation.reachedGoal ? evaluation.steps : -1,
        firstSolvedStep,
        elapsed);
    DestroyLearner(&learner);
    return 0;
}

int RunPpoExperiment(int argc, char **argv)
{
    PpoOptions options;
    if (!ParsePpoOptions(argc, argv, &options)) {
        fprintf(stderr, "Usage: %s --ppo [--steps N] [--seeds N] [--seed N] [--csv FILE] "
            "[--compare-dqn]\n", argv[0]);
        return 2;
    }
    FILE *csv = fopen(options.csvPath, "w");
    if (!csv) {
        fprintf(stderr, "Could not open CSV output: %s\n", options.csvPath);
        return 1;
    }
    fprintf(csv, "algorithm,seed,env_steps,updates,greedy_success,greedy_steps,greedy_return,"
        "mean_episode_return,episodes,goals,entropy,value_loss,clip_fraction,elapsed_ms\n");
    int status = 0;
    for (int offset = 0; offset < options.seeds && status == 0; offset++) {
        uint64_t seed = options.firstSeed + (uint64_t)offset;
        status = RunPpoSeed(csv, &options, seed);
        if (status == 0 && options.compareDqn)
            status = RunDqnBaselineSeed(csv, &options, seed);
        fflush(csv);
    }
    fclose(csv);
    if (status == 0) printf("Wrote PPO results to %s\n", options.csvPath);
    return status;
}

/* ---------- self-tests ---------- */

static bool DistributionSelfTest(void)
{
    float logits[ACTION_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float probabilities[ACTION_COUNT];
    float logProbabilities[ACTION_COUNT];
    PolicyDistribution(logits, probabilities, logProbabilities);
    float sum = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++) {
        sum += probabilities[action];
        if (fabsf(probabilities[action] - 0.25f) > 1.0e-6f) return false;
        if (fabsf(logProbabilities[action] - logf(0.25f)) > 1.0e-5f) return false;
    }
    if (fabsf(sum - 1.0f) > 1.0e-6f) return false;
    /* Uniform over 4 actions has maximum entropy log(4). */
    if (fabsf(PolicyEntropy(probabilities, logProbabilities) - logf(4.0f)) > 1.0e-5f)
        return false;

    /* A large logit should dominate, and probabilities must stay normalized
       even with inputs big enough to overflow a naive exp(). */
    float skewed[ACTION_COUNT] = {100.0f, 0.0f, -100.0f, 0.0f};
    PolicyDistribution(skewed, probabilities, logProbabilities);
    sum = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++) sum += probabilities[action];
    if (fabsf(sum - 1.0f) > 1.0e-5f) return false;
    if (probabilities[0] < 0.99f) return false;
    return true;
}

static bool GaeSelfTest(void)
{
    PpoConfig config = DefaultPpoConfig();
    PpoStep steps[3];
    memset(steps, 0, sizeof(steps));
    steps[0].reward = 1.0f; steps[0].value = 0.5f; steps[0].nextValue = 0.25f;
    steps[1].reward = 2.0f; steps[1].value = 0.25f; steps[1].nextValue = 0.75f;
    steps[2].reward = 3.0f; steps[2].value = 0.75f; steps[2].nextValue = 0.0f;
    steps[2].episodeEnd = true;

    /* lambda = 0 collapses GAE to the one-step TD residual. */
    config.lambda = 0.0f;
    ComputeAdvantages(steps, 3, &config);
    for (int index = 0; index < 3; index++) {
        float delta = steps[index].reward +
            config.gamma * steps[index].nextValue - steps[index].value;
        if (fabsf(steps[index].advantage - delta) > 1.0e-6f) return false;
        /* The critic target is always advantage + value. */
        if (fabsf(steps[index].target - (steps[index].advantage + steps[index].value)) > 1.0e-6f)
            return false;
    }

    /* lambda = 1 makes advantage + value the plain discounted return, since
       this trajectory terminates at the last step. */
    config.lambda = 1.0f;
    ComputeAdvantages(steps, 3, &config);
    float discountedReturn = steps[0].reward +
        config.gamma * steps[1].reward +
        config.gamma * config.gamma * steps[2].reward;
    if (fabsf(steps[0].target - discountedReturn) > 1.0e-5f) return false;

    /* An episode boundary must stop advantage from chaining backwards. */
    memset(steps, 0, sizeof(steps));
    steps[0].reward = 1.0f; steps[0].episodeEnd = true;
    steps[1].reward = 1.0f;
    steps[2].reward = 1.0f;
    config.lambda = 1.0f;
    ComputeAdvantages(steps, 3, &config);
    if (fabsf(steps[0].advantage - 1.0f) > 1.0e-6f) return false;
    return true;
}

static bool ActorGradientSelfTest(void)
{
    Rng rng;
    RngSeed(&rng, UINT64_C(0x9e3779b9));
    ActorNetwork actor;
    InitializeActor(&actor, &rng);
    PpoConfig config = DefaultPpoConfig();

    PpoStep step;
    memset(&step, 0, sizeof(step));
    step.state = 11;
    step.action = ACTION_RIGHT;
    /* Set logProbOld to the policy's current value so the ratio starts at 1,
       safely inside the clip range where the objective is differentiable. */
    float hidden[PPO_HIDDEN];
    float logits[ACTION_COUNT];
    ActorForward(&actor, step.state, hidden, logits);
    float probabilities[ACTION_COUNT];
    float logProbabilities[ACTION_COUNT];
    PolicyDistribution(logits, probabilities, logProbabilities);
    step.logProbOld = logProbabilities[step.action];
    const float advantage = 0.7f;

    ActorNetwork gradient;
    memset(&gradient, 0, sizeof(gradient));
    AccumulateActorGradient(&actor, &step, advantage, &config, 1.0f, &gradient, NULL, NULL);

    const float epsilon = 1.0e-3f;
    /* Check both an output-layer and a hidden-layer weight, so an error in
       either backprop stage is caught. */
    float *probes[2] = {
        &actor.w2[ACTION_RIGHT][0],
        &actor.w1[0][step.state]
    };
    const float *analytic[2] = {
        &gradient.w2[ACTION_RIGHT][0],
        &gradient.w1[0][step.state]
    };
    for (int probe = 0; probe < 2; probe++) {
        float original = *probes[probe];
        *probes[probe] = original + epsilon;
        float plus = ActorLossFor(&actor, &step, advantage, &config);
        *probes[probe] = original - epsilon;
        float minus = ActorLossFor(&actor, &step, advantage, &config);
        *probes[probe] = original;
        float numerical = (plus - minus) / (2.0f * epsilon);
        if (fabsf(numerical - *analytic[probe]) > 2.0e-3f) return false;
    }
    return true;
}

static bool CriticGradientSelfTest(void)
{
    Rng rng;
    RngSeed(&rng, UINT64_C(0x243f6a88));
    CriticNetwork critic;
    InitializeCritic(&critic, &rng);
    const int state = 23;
    const float target = 0.4f;

    CriticNetwork gradient;
    memset(&gradient, 0, sizeof(gradient));
    AccumulateCriticGradient(&critic, state, target, 1.0f, &gradient, NULL);

    const float epsilon = 1.0e-3f;
    float *probes[2] = {&critic.w2[0], &critic.w1[0][state]};
    const float *analytic[2] = {&gradient.w2[0], &gradient.w1[0][state]};
    for (int probe = 0; probe < 2; probe++) {
        float original = *probes[probe];
        *probes[probe] = original + epsilon;
        float plus = CriticLossFor(&critic, state, target);
        *probes[probe] = original - epsilon;
        float minus = CriticLossFor(&critic, state, target);
        *probes[probe] = original;
        float numerical = (plus - minus) / (2.0f * epsilon);
        if (fabsf(numerical - *analytic[probe]) > 2.0e-3f) return false;
    }
    return true;
}

static bool ClipSelfTest(void)
{
    PpoConfig config = DefaultPpoConfig();
    ActorNetwork actor;
    memset(&actor, 0, sizeof(actor));
    PpoStep step;
    memset(&step, 0, sizeof(step));
    step.state = 5;
    step.action = ACTION_UP;
    /* Uniform policy (all-zero weights) gives logProb = log(0.25). Claiming a
       much larger old log-probability drives the ratio far below 1 - eps. */
    step.logProbOld = logf(0.25f) + 2.0f;

    ActorNetwork gradient;
    memset(&gradient, 0, sizeof(gradient));
    bool wasClipped = false;
    /* Negative advantage with a ratio below 1 - eps selects the clipped
       branch, which must contribute no policy gradient. */
    AccumulateActorGradient(&actor, &step, -1.0f, &config, 1.0f, &gradient,
        NULL, &wasClipped);
    if (!wasClipped) return false;

    /* Entropy is the only remaining gradient source, and for a uniform
       policy the entropy gradient is exactly zero, so the output layer
       gradient must vanish entirely here. */
    for (int action = 0; action < ACTION_COUNT; action++) {
        if (fabsf(gradient.b2[action]) > 1.0e-6f) return false;
    }

    /* The same ratio with a positive advantage selects the unclipped
       branch, which must produce a nonzero gradient. */
    memset(&gradient, 0, sizeof(gradient));
    wasClipped = true;
    AccumulateActorGradient(&actor, &step, 1.0f, &config, 1.0f, &gradient,
        NULL, &wasClipped);
    if (wasClipped) return false;
    float magnitude = 0.0f;
    for (int action = 0; action < ACTION_COUNT; action++)
        magnitude += fabsf(gradient.b2[action]);
    if (magnitude < 1.0e-6f) return false;
    return true;
}

static bool DeterminismSelfTest(void)
{
    PpoConfig config = DefaultPpoConfig();
    config.rolloutSteps = 128;
    config.epochs = 1;
    config.minibatchSize = 32;

    PpoAgent *first = malloc(sizeof(PpoAgent));
    PpoAgent *second = malloc(sizeof(PpoAgent));
    if (!first || !second) {
        free(first);
        free(second);
        return false;
    }

    bool equal = true;
    for (int run = 0; run < 2; run++) {
        PpoAgent *agent = run == 0 ? first : second;
        Rng rng;
        RngSeed(&rng, UINT64_C(12345));
        InitializeAgentNetworks(agent, config, &rng);
        CollectRollout(agent, &rng);
        ComputeAdvantages(agent->steps, agent->stepCount, &agent->config);
        NormalizeAdvantages(agent->steps, agent->stepCount);
        TrainOnRollout(agent, &rng, NULL);
    }
    if (memcmp(&first->actor, &second->actor, sizeof(ActorNetwork)) != 0) equal = false;
    if (memcmp(&first->critic, &second->critic, sizeof(CriticNetwork)) != 0) equal = false;
    if (first->environmentSteps != second->environmentSteps) equal = false;

    free(first);
    free(second);
    return equal;
}

bool PpoRunSelfTests(void)
{
    /* Parameter counts: the actor deliberately matches the single-maze DQN
       (6,724) so the comparison is not also a network-size comparison. */
    if (sizeof(ActorNetwork) / sizeof(float) != 6724) return false;
    if (sizeof(CriticNetwork) / sizeof(float) != 6529) return false;
    if (!DistributionSelfTest()) return false;
    if (!GaeSelfTest()) return false;
    if (!ActorGradientSelfTest()) return false;
    if (!CriticGradientSelfTest()) return false;
    if (!ClipSelfTest()) return false;
    if (!DeterminismSelfTest()) return false;
    return true;
}
