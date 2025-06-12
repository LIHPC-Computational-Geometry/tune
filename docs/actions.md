# Triangular Actions

## Flip 

 The **flip** operation flips an inner edge adjacent to two triangles. This selected edge is then deleted and replaced by an edge on the opposite diagonal. The following diagram shows the steps involved in flipping strand d.
 
<img src="img/actions/flip.png" width="300"/>

However, in some configurations, flip action can lead to problematic configurations. That's why we choose to constraint flip action in these situations :

* **Boundary darts :** When a dart is on boundary of the mesh, we should'nt flip the edge. We can check this condition by looking if the dart to flip has a twin dart.
* **Adjacency too high :** When nodes C and D already have an adjacency higher than 10, flip is possible. 
* **Configuration who tends to imply edge reversal**: To detect these situations, we look a the type of quad formed by the two adjacent faces, it means the **quad ADBC**.
    * **concave quad:** When the quad is concave, flip operation necessarily lead to an edge reversal. We constraint it.
Exemple:
  
    * **triangular quad:** Avoid because it will create a flat triangle.
Exemple:

**The only configuration we accept is convexe quads**

