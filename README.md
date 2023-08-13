
<img align="right" width="30%" src="examples/bunny.png">

# GL Mathematics with Numpy

GLM offers a set of objects and functions to ease the design of 3D applications while taking advantage of the numpy library. The main objects are vectors (`vec2`, `vec3`, `vec4`), matrices (`mat2`, `mat3`, `mat4`) and vectorized lists (`vlist`).  Vectors and matrices possess several variants depending on the base type.

#### Example usage

This is the code to get the 3D bunny display on the top-right corner. You'll need  [meshio](https://github.com/nschloe/meshio) to read the mesh file and [matplotlib](https://matplotlib.org/) to display it.

```python
import glm
import numpy as np

# Read mesh, get vertices (vec3) and face indices (int)
import meshio
mesh = meshio.read("bunny.obj")
vertices, indices = mesh.points, mesh.cells[0].data

# Transform: Model / View / Projection matrix (MVP)
MVP = glm.perspective(25, 1, 1, 100) @ glm.translate(0.1, -0.45, -2.5)
MVP = MVP @ glm.xrotate(20) @ glm.yrotate(45) @ glm.scale(5)

# Apply transform
vertices = glm.vec3(glm.vec4(vertices) @ MVP.T)

# Generate faces and sort them
faces = vertices[indices]
faces = faces[np.argsort(-faces[...,2].mean(axis=1))]

# Render faces using matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False, xlim=[-1,+1], ylim=[-1,+1])
ax.add_collection(PolyCollection(faces[...,:2], alpha=0.85, linewidth = 0.5,
                                 facecolor="white", edgecolor="black"))
plt.show()
```

## API

### Vectors & Matrices

The generic notation is:

* `TvecN` for vectors
* `TmatN` for matrices,

with T in [`b`, `i` , `u`, `h`, `d`, `f`, Ø] and N in [2,3,4].

| Symbol | Type             | Size    | Dtype         |
|--------|------------------|---------|---------------|
| `b`    | unsigned integer | 8 bits  | `np.uint8`    |
| `i`    | signed integer   | 32 bits | `np.int32`    |
| `u`    | unsigned integer | 32 bits | `np.uint32`   |
| `h`    | float            | 16 bits | `np.float16`  |
|  Ø     | float            | 32 bits | `np.float32`  |
| `h`    | float            | 64 bits | `np.float64`  |

**Example usage**

```python
import glm

# 10 x 3-components vectors of integers (32 bits)
V = glm.ivec3(10)
V.xyz = 1,2,3

# 10 x 3-components vectors of float (32 bits)
V = glm.vec3(10)

# Conversion from vec3 to vec4 (w set to 1)
V = glm.vec4(V)

# Conversion from vec4 to vec3 (xyz divided by w)
V = glm.vec3(V)

# A 4x4 matrix of floats (32 bits)
M = glm.mat4()
M.xyzw = 1,2,3,1

```

### Vectorized lists 

Vectorized lists (`vlist`) correspond to ragged arrays where items have the same type but different lengths. There are three way to declare a vectorized list.

**Example usage**

```python
import glm

# A list of 3 groups of vectors with size 3,3,4
V = glm.vlist([glm.vec4(3), glm.vec4(3), glm.vec4(3)])

# A list of 3 groups of vectors with size 3,3,4
V = glm.vlist(glm.vec4(10), [3,3,4])

# A list of 5 groups of vectors with size 2
V = glm.vlist(glm.vec4(10), 2)

# Display each item
for v in V: print(v)
```

The underlying structure is a regular numpy array (`vlist.data`).


### Functions
GLM comes with all the standard GL mathematics functions: 

- [viewport](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glViewport.xhtml) (x, y, width, height, *dtype, transpose*)
- [ortho](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml) (left, right, bottom, top, znear, zfar, *dtype, transpose*)
- [frustum](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml) (left, right, bottom, top, znear, zfar, *dtype, transpose*)
- [perspective](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml) (fovy, aspect, znear, zfar, *dtype, transpose*)
- [scale](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glScale.xml) (x, y, z, *dtype, transpose*)
- [rotate](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glRotate.xml) (angle, x, y, z, *dtype, transpose*)
- [translate](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glTranslate.xml) (x, y, z, *dtype, transpose*)
- [lookat](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml) (eye, center, up, *dtype, transpose*)

<!-- 
GlM offers also some convenient functions and classes:
- [trackball](http://scv.bu.edu/documentation/presentations/visualizationworkshop08/materials/opengl/trackball.c) class
-->

