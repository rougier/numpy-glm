"""
Example by: Hana Zupan
An example for displaying scipy spherical Voronoi cells.
"""
import numpy as np
from numpy.typing import ArrayLike
import glm
from glm.camera import Camera
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi
from matplotlib.collections import PolyCollection
import matplotlib as mpl


def random_spherical_point(n_points: int) -> np.ndarray:
    """
    Generate random points that are projected on the surface of a sphere.
    Returns an array in which every row is a 3D point.

    :param n_points: the number of points returned
    :return: an array of shape (n_points, 3)
    """
    vec = np.random.randn(3, n_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def sph_voronoi_random_points(n_points: int, radius: float = 1.0,
                              center: np.ndarray = None) -> SphericalVoronoi:
    """
    Use scipy function to create Voronoi cells on a sphere

    :param n_points: wished number of points
    :param radius: radius of the sphere
    :param center: position of the center of the sphere
    :return: the spherical voronoi with ordered vertices
    """
    if center is None:
        center = np.zeros((3,))
    points = random_spherical_point(n_points)
    sv = SphericalVoronoi(points, radius, center)
    sv.sort_vertices_of_regions()
    return sv

def _generate_faces(vertices: ArrayLike, indexes: ArrayLike) -> tuple:
    """
    Generate faces that may have different number of vertices. Order according
    to the z-value (depth). Each face is a collection of belonging vertices.
    We also return the sorting indices to help keep the color of each face
    consistent.

    :param vertices: each element a 3D vertix
    :param indexes: each element a list of indices belonging to one face
    :return: faces, sorting indices
    """
    faces = [vertices[i] for i in indexes]
    z = [-float(f[..., 2].mean()) for f in faces]
    ind = np.argsort(z).tolist()
    faces = [faces[i] for i in ind]
    return faces, ind



if __name__ == "__main__":
    my_n_points = 100
    my_sv = sph_voronoi_random_points(my_n_points)

    # Make a list of colors of the right length
    mpl.style.use('seaborn')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    my_colors = []
    while len(my_colors) < my_n_points:
        for c in colors:
            my_colors.append(c)

    # generate V, I, F
    vertices = glm.to_vec3(glm.to_vec4(my_sv.vertices))
    indices = my_sv.regions
    vertices = glm.fit(vertices)
    faces, ind = _generate_faces(vertices, indices)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1], aspect=1, frameon=False, xlim=[-1, +1], ylim=[-1, +1])
    my_polygons = PolyCollection([], alpha=0.85, linewidth=0.5, facecolor="white", edgecolor="black")


    def update(transform):
        """
        To make a plot interactive.
        """
        V = glm.to_vec3(glm.to_vec4(vertices) @ transform.T)
        F, ind = _generate_faces(V, indices)
        my_polygons.set_verts([f[..., :2] for f in F])
        my_polygons.set_color([my_colors[i] for i in ind])

    ax.add_collection(my_polygons)
    camera = Camera("perspective", theta=-20, phi=2.5)
    camera.connect(ax, "motion", update)
    update(camera.transform)
    plt.show()
