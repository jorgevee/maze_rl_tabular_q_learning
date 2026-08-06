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

static bool GetPolicyAction(
      int state,
      Action *bestAction)
  {
      float firstValue = qTable[state][ACTION_UP];
      float bestValue = firstValue;

      *bestAction = ACTION_UP;

      bool valuesDiffer = false;

      for (int action = 1; action < ACTION_COUNT; action++)
      {
          float value = qTable[state][action];

          if (value != firstValue)
          {
              valuesDiffer = true;
          }

          if (value > bestValue)
          {
              bestValue = value;
              *bestAction = (Action)action;
          }
      }

      // All equal means this state has no useful learned preference.
      return valuesDiffer;
  }
void DrawPolicy(void)
  {
      static const float dx[ACTION_COUNT] = {
          0.0f, 1.0f, 0.0f, -1.0f
      };

      static const float dy[ACTION_COUNT] = {
          -1.0f, 0.0f, 1.0f, 0.0f
      };

      const Color arrowColor = PURPLE;

      for (int y = 0; y < GRID_HEIGHT; y++)
      {
          for (int x = 0; x < GRID_WIDTH; x++)
          {
              if (maze[y][x] == CELL_WALL ||
                  maze[y][x] == CELL_GOAL)
              {
                  continue;
              }

              Position position = { x, y };
              int state = PositionToState(position);

              Action action;

              if (!GetPolicyAction(state, &action))
              {
                  continue;
              }

              Vector2 center = {
                  MAZE_OFFSET_X + x * CELL_SIZE + CELL_SIZE / 2.0f,
                  MAZE_OFFSET_Y + y * CELL_SIZE + CELL_SIZE / 2.0f
              };

              Vector2 direction = {
                  dx[action],
                  dy[action]
              };

              Vector2 tip = {
                  center.x + direction.x * 20.0f,
                  center.y + direction.y * 20.0f
              };

              Vector2 arrowBase = {
                  tip.x - direction.x * 8.0f,
                  tip.y - direction.y * 8.0f
              };

              Vector2 perpendicular = {
                  -direction.y,
                  direction.x
              };

              Vector2 left = {
                  arrowBase.x + perpendicular.x * 6.0f,
                  arrowBase.y + perpendicular.y * 6.0f
              };

              Vector2 right = {
                  arrowBase.x - perpendicular.x * 6.0f,
                  arrowBase.y - perpendicular.y * 6.0f
              };

              DrawLineEx(center, tip, 3.0f, arrowColor);
              DrawLineEx(tip, left, 3.0f, arrowColor);
              DrawLineEx(tip, right, 3.0f, arrowColor);
          }
      }
  }

/*
The complete function has two passes:

  Pass 1: Find minimum and maximum Q-values
  Pass 2: Convert each value into a color

  Normalization maps values into the range 0.0 through 1.0:

  normalized = (value - minimum) / (maximum - minimum)

*/
void DrawValueHeatmap(void)
  {
      float minimumValue = 0.0f;
      float maximumValue = 0.0f;
      bool foundValue = false;

      // Find the minimum and maximum state values.
      for (int y = 0; y < GRID_HEIGHT; y++)
      {
          for (int x = 0; x < GRID_WIDTH; x++)
          {
              if (maze[y][x] == CELL_WALL ||
                  maze[y][x] == CELL_GOAL)
              {
                  continue;
              }

              Position position = { x, y };
              int state = PositionToState(position);
              float value = GetMaximumQValue(state);

              if (!foundValue)
              {
                  minimumValue = value;
                  maximumValue = value;
                  foundValue = true;
              }
              else
              {
                  if (value < minimumValue)
                  {
                      minimumValue = value;
                  }

                  if (value > maximumValue)
                  {
                      maximumValue = value;
                  }
              }
          }
      }

      if (!foundValue || maximumValue <= minimumValue)
      {
          return;
      }
      for (int y = 0; y < GRID_HEIGHT; y++)
      {
          for (int x = 0; x < GRID_WIDTH; x++)
          {
              if (maze[y][x] == CELL_WALL ||
                  maze[y][x] == CELL_GOAL)
              {
                  continue;
              }

              Position position = { x, y };
              int state = PositionToState(position);

              float value = GetMaximumQValue(state);

              float normalized =
                  (value - minimumValue) /
                  (maximumValue - minimumValue);

              unsigned char red =
                  (unsigned char)(40.0f + 215.0f * normalized);

              unsigned char green =
                  (unsigned char)(80.0f + 120.0f * normalized);

              unsigned char blue =
                  (unsigned char)(220.0f - 180.0f * normalized);

              Color heatColor = {
                  red,
                  green,
                  blue,
                  140
              };

              int screenX =
                  MAZE_OFFSET_X + x * CELL_SIZE;

              int screenY =
                  MAZE_OFFSET_Y + y * CELL_SIZE;

              DrawRectangle(
                  screenX + 1,
                  screenY + 1,
                  CELL_SIZE - 2,
                  CELL_SIZE - 2,
                  heatColor
              );
          }
      }
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

EpisodeResult TrainEpisode(
      int episodeNumber,
      float epsilon,
      float alpha,
      float gamma)
  {
      EpisodeResult episodeResult = {
          .episode = episodeNumber,
          .steps = 0,
          .totalReward = 0.0f,
          .reachedGoal = false
      };

      int state = PositionToState(startPosition);

      for (int step = 0;
           step < MAX_STEPS_PER_EPISODE;
           step++)
      {
          Action action = ChooseAction(state, epsilon);

          StepResult stepResult =
              EnvironmentStep(state, action);

          UpdateQValue(
              state,
              action,
              stepResult.reward,
              stepResult.nextState,
              stepResult.done,
              alpha,
              gamma
          );

          episodeResult.totalReward += stepResult.reward;
        //  step + 1 is important because the loop starts counting at zero, but one completed transition means one step.
          episodeResult.steps = step + 1;

          state = stepResult.nextState;

          if (stepResult.done)
          {
              episodeResult.reachedGoal = true;
              break;
          }
      }

      return episodeResult;
  }