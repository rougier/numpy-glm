# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import numpy as np

def cube():
    """ Cube """

    verts = [(0,  0,  0), ( 1,  0,  0), (1, 0, 1), (0, 0, 1),
             (0,  1,  0), ( 1,  1,  0), (1, 1, 1), (0, 1, 1)]
    faces = [[0,1,2,3],[4, 5, 6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    verts = (np.array(verts,dtype=np.float32)-0.5)/np.sqrt(2)
    faces = np.array(faces, dtype=np.uint32)
    return verts, faces

def sphere(radius=1.0, slices=32, stacks=32):
    slices += 1
    stacks += 1
    n = slices*stacks
    vertices = np.zeros((n,3))
    theta1 = np.repeat(np.linspace(0,     np.pi, stacks, endpoint=True), slices)
    theta2 = np.tile  (np.linspace(0, 2 * np.pi, slices, endpoint=True), stacks)

    vertices[:,1] = np.sin(theta1) * np.cos(theta2) * radius
    vertices[:,2] =                  np.cos(theta1) * radius
    vertices[:,0] = np.sin(theta1) * np.sin(theta2) * radius

    indices = []
    for i in range(stacks-1):
        for j in range(slices-1):
            indices.append(i*(slices) + j        )
            indices.append(i*(slices) + j+1      )
            indices.append(i*(slices) + j+slices+1)

            indices.append(i*(slices) + j+slices+1)
            indices.append(i*(slices) + j+slices  )
            indices.append(i*(slices) + j        )

    indices = np.array(indices)
    indices = indices.reshape(len(indices)//3,3)
    return vertices, indices


def tetrahedron():
    """ Tetrahedron with 4 faces, 6 edges and 4 verts """

    a = 2*np.pi/3
    verts = [ (0,.5,0),
                 (.5*np.cos(0*a), -.25, .5*np.sin(0*a)),
                 (.5*np.cos(1*a), -.25, .5*np.sin(1*a)),
                 (.5*np.cos(2*a), -.25, .5*np.sin(2*a))]
    faces = [(1,2,3), (1,2,0), (2,3,0), (3,1,0)]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)

def octahedron():
    """ Octahedron with 8 faces, 12 edges and 6 verts """

    r = 0.5 * 1/np.sqrt(2)
    verts = [(0,.5,0), (0,-.5,0), (-r,0,-r),
                (r,0,-r), (r,0,r), (-r,0,r)]
    faces = [(2,3,0), (3,4,0), (4,5,0), (5,2,0),
             (3,2,1), (4,3,1), (5,4,1), (2,5,1)]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)

def dodecahedron():
    """ Regular dodecahedron with 12 faces, 30 edges and 20 verts """

    r = (1 + np.sqrt(5)) / 2
    verts = [ (-1, -1, +1), (r, 1/r, 0), (r, -1/r, 0), (-r, 1/r, 0),
                 (-r, -1/r, 0), (0, r,1/r), (0, r, -1/r), (1/r, 0, -r),
                 (-1/r, 0, -r), (0, -r, -1/r), (0, -r, 1/r), (1/r, 0, r),
                 (-1/r, 0, r), (+1, +1, -1), (+1, +1, +1), (-1, +1, -1),
                 (-1, +1, +1), (+1, -1, -1), (+1, -1, +1), (-1, -1, -1) ]
    faces = [ (19, 3, 2), (12, 19, 2), (15, 12, 2),
              (8, 14, 2), (18, 8, 2), (3, 18, 2),
              (20, 5, 4), (9, 20, 4), (16, 9, 4),
              (13, 17, 4), (1, 13, 4), (5, 1, 4),
              (7, 16, 4), (6, 7, 4), (17, 6, 4),
              (6, 15, 2), (7, 6, 2), (14, 7, 2),
              (10, 18, 3), (11, 10, 3), (19, 11, 3),
              (11, 1, 5), (10, 11, 5), (20, 10, 5),
              (20, 9, 8), (10, 20, 8), (18, 10, 8),
              (9, 16, 7), (8, 9, 7), (14, 8, 7),
              (12, 15, 6), (13, 12, 6), (17, 13, 6),
              (13, 1, 11), (12, 13, 11), (19, 12, 11) ]
    # faces = [(19,3,2,15,12), (8,14,2,3,18), (20, 5, 4, 16, 9),
    #          (13, 17, 4, 5, 1), (7, 16, 4, 17, 6), (6, 15, 2, 14, 7),
    #          (10, 18, 3, 19, 11), (11, 1, 5, 20, 10), (20, 9, 8, 18, 10),
    #          (9, 16, 7, 14, 8), (12, 15, 6, 17, 13), (13, 1, 11, 19, 12)]
    verts = np.array(verts, dtype=np.float32)/np.sqrt(3)/2
    faces = np.array(faces, dtype=np.uint32)-1
    return verts, faces

def icosahedron():
    """ Regular icosahedron with 20 faces, 30 edges and 12 verts """

    a = (1 + np.sqrt(5)) / 2
    verts = [ (-1,  a,  0), ( 1,  a,  0), (-1, -a,  0),
              ( 1, -a,  0), ( 0, -1,  a), ( 0,  1,  a),
              ( 0, -1, -a), ( 0,  1, -a), ( a,  0, -1),
              ( a,  0,  1), (-a,  0, -1), (-a,  0,  1) ]
    faces = [ [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
              [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
              [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
              [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1] ]
    verts = np.array(verts, dtype=np.float32)/np.sqrt(a+2)/2
    faces = np.array(faces, dtype=uint32)
    return verts, faces
