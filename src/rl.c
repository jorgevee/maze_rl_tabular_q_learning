#include "rl.h"

const char *AgentKindName(AgentKind kind)
{
    return kind == AGENT_DQN ? "dqn" : "tabular";
}
