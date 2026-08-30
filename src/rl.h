#ifndef RL_H
#define RL_H

#include <stdbool.h>

typedef enum {
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_COUNT
} Action;

typedef enum {
    AGENT_TABULAR,
    AGENT_DQN,
    AGENT_KIND_COUNT
} AgentKind;

typedef struct {
    int state;
    Action action;
    float reward;
    int nextState;
    bool done;
} Transition;

typedef struct {
    int episode;
    int steps;
    float totalReward;
    bool reachedGoal;
} EpisodeResult;

const char *AgentKindName(AgentKind kind);

#endif
