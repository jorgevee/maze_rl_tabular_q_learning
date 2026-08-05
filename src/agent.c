#include "agent.h"
#include "maze.h"
#include "raylib.h"
#include "environment.h"
//Add defenitions and initializations for the extern variables declared in agent.h
Position startPosition = {0, 0};
Position agentPosition = {0, 0};
Position goalPosition = {0, 0};
float qTable[STATE_COUNT][ACTION_COUNT];

void InitializeAgent(void)
{
    for (int y = 0; y < GRID_HEIGHT; y++)
    {
        for (int x = 0; x < GRID_WIDTH; x++)
        {
            if (maze[y][x] == CELL_START)
            {
                startPosition.x = x;
                startPosition.y = y;
                // agentPosition.x = x;
                // agentPosition.y = y;
            }
            else if (maze[y][x] == CELL_GOAL)
            {
                goalPosition.x = x;
                goalPosition.y = y;
            }
        }
    }
    ResetAgent();
}

void ResetAgent(void)
{
    agentPosition = startPosition;
}


bool TryMove(Position *position, Action action)
  {
      static const int dx[ACTION_COUNT] = { 0, 1, 0, -1 };
      static const int dy[ACTION_COUNT] = { -1, 0, 1, 0 };

      int targetX = position->x + dx[action];
      int targetY = position->y + dy[action];

      if (targetX < 0 || targetX >= GRID_WIDTH ||
          targetY < 0 || targetY >= GRID_HEIGHT)
      {
          return false;
      }

      if (maze[targetY][targetX] == CELL_WALL)
          return false;

      position->x = targetX;
      position->y = targetY;
      return true;
  }

void DrawAgent(Position position)
  {
      int centerX =
          MAZE_OFFSET_X + position.x * CELL_SIZE + CELL_SIZE / 2;

      int centerY =
          MAZE_OFFSET_Y + position.y * CELL_SIZE + CELL_SIZE / 2;

      DrawCircle(
          centerX,
          centerY,
          CELL_SIZE / 3,
          RED
      );
}
// Every Q-value begins at zero because the agent has not learned anything yet.
void InitializeQTable(void)
  {
      for (int state = 0; state < STATE_COUNT; state++)
      {
          for (int action = 0; action < ACTION_COUNT; action++)
          {
              qTable[state][action] = 0.0f;
          }
      }
  }

/*
  Greedy action selection means:

  Look at all four Q-values for the current state and select an action with the highest value.
*/
  Action GetBestAction(int state)
  {
      Action bestActions[ACTION_COUNT];
      int bestActionCount = 0;

      float bestValue = qTable[state][ACTION_UP];

      for (int action = 0; action < ACTION_COUNT; action++)
      {
          float value = qTable[state][action];

          if (value > bestValue)
          {
              // We found a new highest value.
              bestValue = value;
              bestActionCount = 0;

              bestActions[bestActionCount] = (Action)action;
              bestActionCount++;
          }
          else if (value == bestValue)
          {
              // This action is tied for the highest value.
              bestActions[bestActionCount] = (Action)action;
              bestActionCount++;
          }
      }

      int selectedIndex =
          GetRandomValue(0, bestActionCount - 1);

      return bestActions[selectedIndex];
  }

// Generate a random probability
static float RandomFloat(void)
  {
      return (float)GetRandomValue(0, 9999) / 10000.0f;
  }
// Implement epsilon-greedy selection

/*
The decision looks like:

  Generate random number
            ↓
  Is it below epsilon?
         /       \
       yes        no
        ↓          ↓
   random action  best-known action

  For example, with:

  epsilon = 0.20f;

  approximately:

  20% → random action
  80% → greedy action
  
  */
Action ChooseAction(int state, float epsilon)
  {
    if (RandomFloat() < epsilon)
    {
        // Explore: ignore the Q-values and try any action.
        return (Action)GetRandomValue(0, ACTION_COUNT - 1);
    }

    // Exploit: use the best action currently known.
    return GetBestAction(state);
}
//  Find the maximum future Q-value
float GetMaximumQValue(int state)
  {
      float maximum = qTable[state][ACTION_UP];

      for (int action = 1; action < ACTION_COUNT; action++)
      {
          if (qTable[state][action] > maximum)
          {
              maximum = qTable[state][action];
          }
      }

      return maximum;
  }

   void UpdateQValue(
      int state,
      Action action,
      float reward,
      int nextState,
      bool done,
      float alpha,
      float gamma)
  {
      float currentQ = qTable[state][action];

      float futureQ = 0.0f;

      if (!done)
      {
          futureQ = GetMaximumQValue(nextState);
      }

      float target = reward + gamma * futureQ;
      float error = target - currentQ;

      qTable[state][action] += alpha * error;
  }

float TrainEpisode(
      float epsilon,
      float alpha,
      float gamma)
  {
      int state = PositionToState(startPosition);
      float totalReward = 0.0f;

      for (int step = 0;
           step < MAX_STEPS_PER_EPISODE;
           step++)
      {
          // 1. Select an action.
          Action action = ChooseAction(state, epsilon);

          // 2. Apply the action to the environment.
          StepResult result =
              EnvironmentStep(state, action);

          // 3. Learn from the transition.
          UpdateQValue(
              state,
              action,
              result.reward,
              result.nextState,
              result.done,
              alpha,
              gamma
          );

          // 4. Record the reward and advance the state.
          totalReward += result.reward;
          state = result.nextState;

          // 5. Stop if the goal was reached.
          if (result.done)
          {
              break;
          }
      }

      return totalReward;
  }