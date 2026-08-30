#include "tabular.h"
#include "maze.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    float q[STATE_COUNT][ACTION_COUNT];
    float alpha;
    float gamma;
} TabularState;

static void Destroy(void *state) { free(state); }

static void Reset(void *state, Rng *rng)
{
    (void)rng;
    TabularState *table = state;
    memset(table->q, 0, sizeof(table->q));
}

static void Values(const void *state, int mazeState, float values[ACTION_COUNT])
{
    const TabularState *table = state;
    memcpy(values, table->q[mazeState], sizeof(table->q[mazeState]));
}

static Action Select(void *state, int mazeState, float epsilon, Rng *rng)
{
    TabularState *table = state;
    if (RngFloat(rng) < epsilon) return (Action)RngRange(rng, ACTION_COUNT);

    Action best[ACTION_COUNT];
    int count = 1;
    float maximum = table->q[mazeState][0];
    best[0] = ACTION_UP;
    for (int action = 1; action < ACTION_COUNT; action++) {
        float value = table->q[mazeState][action];
        if (value > maximum) {
            maximum = value;
            count = 1;
            best[0] = (Action)action;
        } else if (value == maximum) {
            best[count++] = (Action)action;
        }
    }
    return best[RngRange(rng, count)];
}

static void Observe(void *state, Transition transition, Rng *rng)
{
    (void)rng;
    TabularState *table = state;
    float future = 0.0f;
    if (!transition.done) {
        future = table->q[transition.nextState][0];
        for (int action = 1; action < ACTION_COUNT; action++) {
            if (table->q[transition.nextState][action] > future)
                future = table->q[transition.nextState][action];
        }
    }
    float *current = &table->q[transition.state][transition.action];
    *current += table->alpha * (transition.reward + table->gamma * future - *current);
}

static size_t Memory(const void *state) { (void)state; return sizeof(TabularState); }
static size_t Parameters(const void *state) { (void)state; return STATE_COUNT * ACTION_COUNT; }

bool CreateTabularLearner(Learner *learner, float alpha, float gamma)
{
    TabularState *state = calloc(1, sizeof(*state));
    if (state == NULL) return false;
    state->alpha = alpha;
    state->gamma = gamma;
    learner->kind = AGENT_TABULAR;
    learner->state = state;
    learner->ops = (LearnerOps){ Destroy, Reset, Select, Observe, Values, Memory, Parameters };
    return true;
}
