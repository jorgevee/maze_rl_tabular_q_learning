 #ifndef AGENT_H
 #define AGENT_H

#include <stdbool.h>
#include "maze.h"


  typedef enum {
      ACTION_UP,
      ACTION_RIGHT,
      ACTION_DOWN,
      ACTION_LEFT,
      ACTION_COUNT
  } Action;

  extern Position startPosition;
  extern Position agentPosition;
  extern Position goalPosition;

  void ResetAgent(void);
  bool TryMove(Position *position, Action action);
  void InitializeAgent(void);
  void DrawAgent(Position position);
  
  /*
Declare Q-table.
For example:

qTable[11][ACTION_RIGHT]

means:
How valuable does the agent currently think moving right from state 11 is?
*/
extern float qTable[STATE_COUNT][ACTION_COUNT];
void InitializeQTable(void);

Action GetBestAction(int state);
// Epsilon-greedy selection
Action ChooseAction(int state, float epsilon);
// After seeing the reward and next state, how should the agent revise the value of the action it just took?
float GetMaximumQValue(int state);

  void UpdateQValue(
      int state,
      Action action,
      float reward,
      int nextState,
      bool done,
      float alpha,
      float gamma
);

// We return float because rewards are stored as floating-point values.
float TrainEpisode(
      float epsilon,
      float alpha,
      float gamma
  );
  #endif