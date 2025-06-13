# Triangular Actions

## Flip 

 The **flip** operation flips an inner edge adjacent to two triangles. This selected edge is then deleted and replaced by an edge on the opposite diagonal. The following diagram shows the steps involved in flipping strand d.
 
<img src="img/actions/flip.png" width="300"/>

However, in some configurations, flip action can lead to problematic configurations. That's why we choose to constraint flip action in these situations :

* **Boundary darts :** When a dart is on boundary of the mesh, we should'nt flip the edge. We can check this condition by looking if the dart to flip has a twin dart.
* **Adjacency too high :** When nodes C and D already have an adjacency higher than 10, flip is not possible. 
* **Configuration who tends to imply edge reversal**: To detect these situations, we look a the type of quad formed by the two adjacent faces, i.e. **quad ADBC**.
    * **concave quad:** When the quad is concave, flip operation necessarily lead to an edge reversal. We constraint it.
  
  <img src="img/actions/flip_c_bef.png" width="200"/>
  <img src="img/actions/flip_c_after.png" width="200"/>

    * **triangular quad:** Avoid because it will create a flat triangle.
  
  <img src="img/actions/flip_before_tri.png" width="200"/>
  <img src="img/actions/flip_after_tri.png" width="200"/>

**The only configuration we accept is convexe quads**

## Split

 The **split** operation split an inner edge adjacent to two triangles. A node is added in the middle of the edge and two faces are created.

<img src="img/actions/split.png" width="300"/>

However, in some configurations, split action can also lead to problematic configurations. That's why we choose to constraint split action in these situations :

* **Boundary darts :** When a dart is on boundary of the mesh, we decide to prohibit it. It can be possible if we add another boundary node, but we choose to not touch mesh boundaries. We can check this condition by looking if the dart to split has a twin dart.
* **Adjacency too high :** When nodes C and D already have an adjacency higher than 10, split is not possible. 
* **Configuration who tends to imply null darts**: As other actions are restricted to not allow flat faces, or reversed faces, split action can be performed on each configuration, see figure xxx.

<img src="img/actions/split_concave.png" width="300"/>
<img src="img/actions/split_triangular.png" width="300"/>

## Collapse

The **collapse** operation deletes an inner edge and also implies the deletion of its two adjacent faces F1 and F2.

<img src="img/actions/split.png" width="300"/>

However, in some configurations, collapse action can also lead to problematic configurations. That's why we choose to constraint collapse action in these situations :

* **Boundary darts :** When a dart is on boundary of the mesh, we decide to prohibit it. We choose to not touch mesh boundaries. We can check this condition by looking if the dart to split has a twin dart.
* **Adjacency too high :** When nodes A already has an adjacency higher than 10, collapse is not possible. 
* **Configuration who tends to imply edge reversal**: To detect these situations, we look a the type of darts in the surrounding. When there are some darts with concave surrounding, collapse action can lead to edge reversal.


