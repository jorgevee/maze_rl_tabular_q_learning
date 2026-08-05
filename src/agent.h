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

// Declare Q-table
extern float qTable[STATE_COUNT][ACTION_COUNT];
void InitializeQTable(void);

  #endif