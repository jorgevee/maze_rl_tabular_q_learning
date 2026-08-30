#ifndef AGENT_H
#define AGENT_H

#include <stdbool.h>
#include "maze.h"
#include "rl.h"
#include "learner.h"

extern Position startPosition;
extern Position agentPosition;
extern Position goalPosition;

void InitializeAgent(void);
void ResetAgent(void);
bool TryMove(Position *position, Action action);
void DrawAgent(Position position);
void DrawPolicy(const Learner *learner);
void DrawValueHeatmap(const Learner *learner);

#endif
