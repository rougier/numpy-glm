# Quickstart

#### 1. Read mesh

   ```python
   import meshio
   mesh = meshio.read("bunny.obj")
   vertices, indices = mesh.points, mesh.cells[0].data
   ```

#### 2. Create transform

  ```python
  import glm
  MVP = glm.perspective(25, 1, 1, 100) @ glm.translate((0.1, -0.45, -2.5))
  MVP = MVP @ glm.xrotate(20) @ glm.yrotate(45) @ glm.scale((5,5,5))
  ```

#### 3. Apply transform

  ```python
  vertices = glm.to_vec3(glm.to_vec4(vertices) @ M.T)
  ```


#### 4. Generate and sort faces

  ```python
  import numpy as np
  faces = vertices[indices]
  faces = faces[np.argsort(-faces[...,2].mean(axis=1))]
  ```
#### 5. Render

  ```python
  import matplotlib.pyplot as plt
  from matplotlib.collections import PolyCollection
  fig = plt.figure(figsize=(6,6))
  ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False, xlim=[-1,+1], ylim=[-1,+1])
  ax.add_collection(PolyCollection(faces[...,:2], alpha=0.85, linewidth = 0.5,
                                   facecolor="white", edgecolor="black"))
  plt.show()
  ```
