# What to do in this project?

# Triangular mesh adaptation
In order to learn how to modify a 2D mesh, we first implement a 2D game
that works on pure triangular meshes. Starting from a 2D mesh, the goal 
of the player is to modify the mesh by applying topological operations 
in order to improve the **mesh quality**. Here the quality is based 
on topological criteria only. More precisely, we consider the degree 
of nodes. The degree of a node is defined as its number of adjacent 
faces. Ideally:
- The degree of an inner node is 6,
- The degree of a boundary node *n* depends on the local geometry 
around *n*.

### Version 1 - Triangles and edge flip
In the first version, we have only one 
operation - the *edge flipping* - and we provide triangular meshes for which we know
what the best solution is. The aim of the intelligent agent we develop is to build this best 
solution in a mininum number of movements.


