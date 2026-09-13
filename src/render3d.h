#ifndef RENDER3D_H
#define RENDER3D_H

/* Interactive 3D view of the maze suite: the same 2D grid the agents
   actually solve, drawn with extruded walls and a perspective camera.

   The environment is unchanged -- the agent still occupies one cell of a
   flat grid and picks one of four moves. Only the presentation is 3D. */
int RunRender3D(int argc, char **argv);

#endif
