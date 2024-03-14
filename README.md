# Topologic UntaNgling mEsher

The aim of this project is to provide an environment to implement 
Reinforcement Learning algorithms that aim to topologically
modify a 2D mesh. More specifically, we implement the work of 
*A. Narayanana*, *Y. Pan*, and *P.-O. Persson*, which is described 
in "**Learning topological operations on meshes with application to 
block decomposition of polygons**" (see [paper]()).

## Incremental game design
In order to learn how to modify a 2D mesh, we first implement a 2D game
that works on pure triangular meshes. Starting from a 2D mesh, the goal 
of the player is to modify the mesh by applying topological operations 
in order to improve the **mesh quality**. Here the quality is defined 
on topological criteria only. More precisely, we consider the degree 
of nodes. The degree of a node is defined as its number of adjacent 
faces. Ideally:
- The degree of an inner node is 6,
- The degree of a boundary node *n* depends on the local geometry 
around *n*.

### Version 1 - Triangles and edge swap
In this first version, we consider a pure triangular mesh, and we have only one 
operation, the *edge swapping*

<img src="./scheme.png" width="70%" height="70%"/>
