# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import glm
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Read mesh and get vertices (vec3) and face indices (int)
mesh = meshio.read("bunny.obj")
vertices, indices = mesh.points, mesh.cells[0].data

# Model / View / Projection matrix (MVP)
M = glm.perspective(25, 1, 1, 100) @ glm.translate((0.1, -0.45, -2.5))
M = M @glm.xrotate(20) @ glm.yrotate(45) @ glm.scale((5,5,5))

# Apply MVP 
vertices = glm.to_vec3(glm.to_vec4(vertices) @ M.T)

# Generate faces and sort them
faces = vertices[indices]
faces = faces[np.argsort(-faces[...,2].mean(axis=1))]

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False,
                  xlim=[-1,+1], ylim=[-1,+1])
ax.add_collection(
     PolyCollection(faces[...,:2], alpha=0.85, linewidth = 0.5,
                    facecolor="white", edgecolor="black"))
plt.show()
