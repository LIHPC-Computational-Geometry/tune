# Actions tests

## Triangular actions

### Flip

We want to ensure that **flip** operations are performed correctly.
Consider the following mesh:

<img src="img/tri_flip_before.png" width="300"/>

If we flip the dart between nodes **n0** and **n3**, we should obtain the following mesh:

<img src="img/tri_flip_after.png" width="300"/>

### Split

We want to ensure that **split** operations are performed correctly.
Consider the following mesh:

<img src="img/tri_split_before.png" width="300"/>

If we split the dart between nodes **n0** and **n3**, we add a node **n5** on the middle of the edge and two faces are created. 
We should obtain the following mesh:

<img src="img/tri_split_after.png" width="300"/>

### Collapse

We want to ensure that **split** operations are performed correctly.
Consider the following mesh:

<img src="img/tri_collapse_before.png" width="300"/>

Here we can't collapse the dart between nodes **n5** and **n2** because n2 is on boundary. However we can collapse the edge between **n4** and **n5**.
We should obtain the following mesh:

<img src="img/tri_collapse_after.png" width="300"/>