# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import glm
import meshio
import numpy as np
from glm.camera import Camera
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

mesh = meshio.read("bunny.obj")
vertices, indices = mesh.points, mesh.cells[0].data
vertices = glm.fit(vertices)

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False, xlim=[-1,+1], ylim=[-1,+1])
bunny = PolyCollection([], alpha=0.85, linewidth = 0.5, facecolor="white", edgecolor="black")

def update(transform):
    V = glm.to_vec3(glm.to_vec4(vertices) @ transform.T)
    F = V[indices]
    F = F[np.argsort(-F[...,2].mean(axis=1))]
    bunny.set_verts(F[...,:2])

ax.add_collection(bunny)
camera = glm.Camera("perspective", theta=-20, phi=2.5)
camera.connect(ax, "motion",  update)
update(camera.transform)
plt.show()
