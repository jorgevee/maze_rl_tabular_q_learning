#ifndef MAZE_H
#define MAZE_H

#define GRID_WIDTH 10
#define GRID_HEIGHT 10

#define CELL_SIZE 70
#define MAZE_OFFSET_X 50
#define MAZE_OFFSET_Y 70

#define STATE_COUNT (GRID_WIDTH * GRID_HEIGHT)

  typedef struct {
      int x;
      int y;
  } Position;


typedef enum {
      CELL_EMPTY,
      CELL_WALL,
      CELL_START,
      CELL_GOAL
  } CellType;

 // Declares that the maze exists somewhere.
extern CellType maze[GRID_HEIGHT][GRID_WIDTH];

int PositionToState(Position position);
Position StateToPosition(int state);
void DrawMaze(void);

#endif