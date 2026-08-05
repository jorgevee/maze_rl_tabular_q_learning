 #include "maze.h"
  #include "raylib.h"

  CellType maze[GRID_HEIGHT][GRID_WIDTH] = {
      {1,1,1,1,1,1,1,1,1,1},
      {1,2,0,0,1,0,0,0,0,1},
      {1,0,1,0,1,0,1,1,0,1},
      {1,0,1,0,0,0,0,1,0,1},
      {1,0,1,1,1,1,0,1,0,1},
      {1,0,0,0,0,1,0,0,0,1},
      {1,1,1,1,0,1,1,1,0,1},
      {1,0,0,0,0,0,0,1,0,1},
      {1,0,1,1,1,1,0,0,3,1},
      {1,1,1,1,1,1,1,1,1,1}
  };

  void DrawMaze(void)
  {
      for (int y = 0; y < GRID_HEIGHT; y++)
      {
          for (int x = 0; x < GRID_WIDTH; x++)
          {
              Color color;

              switch (maze[y][x])
              {
                  case CELL_WALL:
                      color = DARKGRAY;
                      break;

                  case CELL_START:
                      color = BLUE;
                      break;

                  case CELL_GOAL:
                      color = GREEN;
                      break;

                  case CELL_EMPTY:
                  default:
                      color = RAYWHITE;
                      break;
              }

              int screenX = MAZE_OFFSET_X + x * CELL_SIZE;
              int screenY = MAZE_OFFSET_Y + y * CELL_SIZE;

              DrawRectangle(
                  screenX,
                  screenY,
                  CELL_SIZE,
                  CELL_SIZE,
                  color
              );

              DrawRectangleLines(
                  screenX,
                  screenY,
                  CELL_SIZE,
                  CELL_SIZE,
                  LIGHTGRAY
              );
          }
      }
  }


/*
The reverse conversion uses:

  - % to find the column (x)
  - / to find the row (y)

  For state 37:

  x = 37 % 10 = 7
  y = 37 / 10 = 3

  state 37 → position (7, 3)

*/

int PositionToState(Position position) {
    return position.y * GRID_WIDTH + position.x;
}

Position StateToPosition(int state) {
    Position position;
    position.x = state % GRID_WIDTH;
    position.y = state / GRID_WIDTH;
    return position;
}