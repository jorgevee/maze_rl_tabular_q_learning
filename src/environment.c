#include "environment.h"
#include "maze.h"
#include "agent.h"

StepResult EnvironmentStep(int state, Action action)
  {
    StepResult result;
     // 1. Convert the integer state into a grid position.
    Position position = StateToPosition(state);

    // 2. Attempt the requested movement.
    bool moved = TryMove(&position, action);

    // 3. Convert the resulting position back into a state.
    result.nextState = PositionToState(position);

    // 4 and 5. Calculate the reward and termination status.
    if (position.x == goalPosition.x &&
        position.y == goalPosition.y)
    {
        result.reward = 100.0f;
        result.done = true;
    }
    else if (!moved)
    {
        result.reward = -5.0f;
        result.done = false;
    }
    else
    {
        result.reward = -1.0f;
        result.done = false;
    }
     return result;
}