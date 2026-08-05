  #include "agent.h"
  #include "maze.h"
  #include "raylib.h"

//Add defenitions and initializations for the extern variables declared in agent.h
  Position startPosition = {0, 0};
  Position agentPosition = {0, 0};
  Position goalPosition = {0, 0};

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