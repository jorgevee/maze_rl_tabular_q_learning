#include "dqn.h"
#include "maze.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN_SIZE 64
#define MAX_REPLAY_CAPACITY 10000
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPSILON 1.0e-8f

typedef struct {
    float w1[HIDDEN_SIZE][STATE_COUNT];
    float b1[HIDDEN_SIZE];
    float w2[ACTION_COUNT][HIDDEN_SIZE];
    float b2[ACTION_COUNT];
} Network;

typedef struct {
    Transition entries[MAX_REPLAY_CAPACITY];
    int capacity;
    int count;
    int next;
} ReplayBuffer;

typedef struct {
    Network online;
    Network target;
    Network firstMoment;
    Network secondMoment;
    ReplayBuffer replay;
    DqnConfig config;
    int optimizerUpdates;
} DqnState;

static void ZeroNetwork(Network *network)
{
    memset(network, 0, sizeof(*network));
}

static void CopyNetwork(Network *destination, const Network *source)
{
    memcpy(destination, source, sizeof(*destination));
}

static void InitializeNetwork(Network *network, Rng *rng)
{
    ZeroNetwork(network);
    const float firstScale = sqrtf(2.0f / STATE_COUNT);
    const float secondScale = sqrtf(2.0f / HIDDEN_SIZE);
    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        for (int input = 0; input < STATE_COUNT; input++)
            network->w1[hidden][input] = RngNormal(rng) * firstScale;
    }
    for (int action = 0; action < ACTION_COUNT; action++) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++)
            network->w2[action][hidden] = RngNormal(rng) * secondScale;
    }
}

static void Forward(
    const Network *network,
    int state,
    float hiddenValues[HIDDEN_SIZE],
    float qValues[ACTION_COUNT])
{
    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        float value = network->w1[hidden][state] + network->b1[hidden];
        hiddenValues[hidden] = value > 0.0f ? value : 0.0f;
    }
    for (int action = 0; action < ACTION_COUNT; action++) {
        float value = network->b2[action];
        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++)
            value += network->w2[action][hidden] * hiddenValues[hidden];
        qValues[action] = value;
    }
}

static float Maximum(const float values[ACTION_COUNT])
{
    float maximum = values[0];
    for (int action = 1; action < ACTION_COUNT; action++)
        if (values[action] > maximum) maximum = values[action];
    return maximum;
}

static float TargetFor(const DqnState *dqn, Transition transition)
{
    if (transition.done) return transition.reward;
    float hidden[HIDDEN_SIZE];
    float nextValues[ACTION_COUNT];
    Forward(&dqn->target, transition.nextState, hidden, nextValues);
    return transition.reward + dqn->config.gamma * Maximum(nextValues);
}

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

static void AccumulateGradient(
    const Network *network,
    int state,
    Action action,
    float target,
    float scale,
    Network *gradient)
{
    float hidden[HIDDEN_SIZE];
    float qValues[ACTION_COUNT];
    Forward(network, state, hidden, qValues);
    float outputGradient = HuberDerivative(qValues[action] - target) * scale;

    gradient->b2[action] += outputGradient;
    for (int index = 0; index < HIDDEN_SIZE; index++) {
        gradient->w2[action][index] += outputGradient * hidden[index];
        if (hidden[index] > 0.0f) {
            float hiddenGradient = outputGradient * network->w2[action][index];
            gradient->b1[index] += hiddenGradient;
            gradient->w1[index][state] += hiddenGradient;
        }
    }
}

static float GradientNorm(const Network *gradient)
{
    const float *values = (const float *)gradient;
    size_t count = sizeof(*gradient) / sizeof(float);
    double sum = 0.0;
    for (size_t index = 0; index < count; index++)
        sum += (double)values[index] * values[index];
    return (float)sqrt(sum);
}

static void ScaleGradient(Network *gradient, float scale)
{
    float *values = (float *)gradient;
    size_t count = sizeof(*gradient) / sizeof(float);
    for (size_t index = 0; index < count; index++) values[index] *= scale;
}

static void AdamUpdate(DqnState *dqn, Network *gradient)
{
    float norm = GradientNorm(gradient);
    if (norm > dqn->config.gradientClip)
        ScaleGradient(gradient, dqn->config.gradientClip / norm);

    dqn->optimizerUpdates++;
    float correction1 = 1.0f - powf(ADAM_BETA1, (float)dqn->optimizerUpdates);
    float correction2 = 1.0f - powf(ADAM_BETA2, (float)dqn->optimizerUpdates);
    float *parameter = (float *)&dqn->online;
    float *first = (float *)&dqn->firstMoment;
    float *second = (float *)&dqn->secondMoment;
    float *grad = (float *)gradient;
    size_t count = sizeof(Network) / sizeof(float);

    for (size_t index = 0; index < count; index++) {
        first[index] = ADAM_BETA1 * first[index] + (1.0f - ADAM_BETA1) * grad[index];
        second[index] = ADAM_BETA2 * second[index] + (1.0f - ADAM_BETA2) * grad[index] * grad[index];
        float firstHat = first[index] / correction1;
        float secondHat = second[index] / correction2;
        parameter[index] -= dqn->config.learningRate * firstHat /
            (sqrtf(secondHat) + ADAM_EPSILON);
    }
}

static void ReplayPush(ReplayBuffer *replay, Transition transition)
{
    replay->entries[replay->next] = transition;
    replay->next = (replay->next + 1) % replay->capacity;
    if (replay->count < replay->capacity) replay->count++;
}

static void TrainBatch(DqnState *dqn, Rng *rng)
{
    Network gradient;
    ZeroNetwork(&gradient);
    const float scale = 1.0f / dqn->config.batchSize;
    for (int sample = 0; sample < dqn->config.batchSize; sample++) {
        Transition transition = dqn->replay.entries[RngRange(rng, dqn->replay.count)];
        AccumulateGradient(
            &dqn->online,
            transition.state,
            transition.action,
            TargetFor(dqn, transition),
            scale,
            &gradient);
    }
    AdamUpdate(dqn, &gradient);
    if (dqn->optimizerUpdates % dqn->config.targetSyncUpdates == 0)
        CopyNetwork(&dqn->target, &dqn->online);
}

static void Destroy(void *state) { free(state); }

static void Reset(void *state, Rng *rng)
{
    DqnState *dqn = state;
    InitializeNetwork(&dqn->online, rng);
    CopyNetwork(&dqn->target, &dqn->online);
    ZeroNetwork(&dqn->firstMoment);
    ZeroNetwork(&dqn->secondMoment);
    dqn->replay.count = 0;
    dqn->replay.next = 0;
    dqn->optimizerUpdates = 0;
}

static void Values(const void *state, int mazeState, float values[ACTION_COUNT])
{
    const DqnState *dqn = state;
    float hidden[HIDDEN_SIZE];
    Forward(&dqn->online, mazeState, hidden, values);
}

static Action Select(void *state, int mazeState, float epsilon, Rng *rng)
{
    if (RngFloat(rng) < epsilon) return (Action)RngRange(rng, ACTION_COUNT);
    float values[ACTION_COUNT];
    Values(state, mazeState, values);
    Action best[ACTION_COUNT];
    int count = 1;
    float maximum = values[0];
    best[0] = ACTION_UP;
    for (int action = 1; action < ACTION_COUNT; action++) {
        if (values[action] > maximum) {
            maximum = values[action];
            count = 1;
            best[0] = (Action)action;
        } else if (values[action] == maximum) {
            best[count++] = (Action)action;
        }
    }
    return best[RngRange(rng, count)];
}

static void Observe(void *state, Transition transition, Rng *rng)
{
    DqnState *dqn = state;
    ReplayPush(&dqn->replay, transition);
    if (dqn->replay.count >= dqn->config.replayWarmup) TrainBatch(dqn, rng);
}

static size_t Memory(const void *state) { (void)state; return sizeof(DqnState); }
static size_t Parameters(const void *state)
{
    (void)state;
    return sizeof(Network) / sizeof(float);
}

DqnConfig DefaultDqnConfig(void)
{
    return (DqnConfig){
        .gamma = 0.95f,
        .learningRate = 0.001f,
        .replayCapacity = MAX_REPLAY_CAPACITY,
        .replayWarmup = 500,
        .batchSize = 32,
        .targetSyncUpdates = 250,
        .gradientClip = 10.0f
    };
}

bool CreateDqnLearner(Learner *learner, DqnConfig config, Rng *rng)
{
    if (config.replayCapacity < 1 || config.replayCapacity > MAX_REPLAY_CAPACITY ||
        config.replayWarmup < 1 || config.replayWarmup > config.replayCapacity ||
        config.batchSize < 1 || config.targetSyncUpdates < 1 ||
        config.learningRate <= 0.0f || config.gradientClip <= 0.0f)
        return false;

    DqnState *state = calloc(1, sizeof(*state));
    if (state == NULL) return false;
    state->config = config;
    state->replay.capacity = config.replayCapacity;
    learner->kind = AGENT_DQN;
    learner->state = state;
    learner->ops = (LearnerOps){ Destroy, Reset, Select, Observe, Values, Memory, Parameters };
    Reset(state, rng);
    return true;
}

static float LossFor(const Network *network, int state, Action action, float target)
{
    float hidden[HIDDEN_SIZE];
    float values[ACTION_COUNT];
    Forward(network, state, hidden, values);
    return HuberLoss(values[action] - target);
}

bool DqnRunSelfTests(void)
{
    ReplayBuffer replay = { .capacity = 3 };
    for (int value = 0; value < 5; value++) {
        Transition transition = { .state = value };
        ReplayPush(&replay, transition);
    }
    if (replay.count != 3 || replay.next != 2 || replay.entries[1].state != 4) return false;

    Network network;
    ZeroNetwork(&network);
    network.w1[0][11] = 0.7f;
    network.w2[ACTION_RIGHT][0] = 0.4f;
    Network gradient;
    ZeroNetwork(&gradient);
    const float target = 0.1f;
    AccumulateGradient(&network, 11, ACTION_RIGHT, target, 1.0f, &gradient);
    const float epsilon = 1.0e-3f;
    network.w2[ACTION_RIGHT][0] += epsilon;
    float plus = LossFor(&network, 11, ACTION_RIGHT, target);
    network.w2[ACTION_RIGHT][0] -= 2.0f * epsilon;
    float minus = LossFor(&network, 11, ACTION_RIGHT, target);
    float numerical = (plus - minus) / (2.0f * epsilon);
    if (fabsf(numerical - gradient.w2[ACTION_RIGHT][0]) > 1.0e-3f) return false;

    DqnState dqn;
    memset(&dqn, 0, sizeof(dqn));
    dqn.config.gamma = 0.95f;
    Transition terminal = { .reward = 100.0f, .done = true };
    if (TargetFor(&dqn, terminal) != 100.0f) return false;
    CopyNetwork(&dqn.target, &network);
    if (memcmp(&dqn.target, &network, sizeof(network)) != 0) return false;
    return true;
}
