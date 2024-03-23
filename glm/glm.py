# -----------------------------------------------------------------------------
# GL Mathematics for numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import numpy as np
from . ndarray import mat4

def mesh(filename):
    """
    Read a wavefront filename and returns vertices, texcoords and
    respective indices for faces and texcoords
    """

    V, T, N, Vi, Ti, Ni = [], [], [], [], [], []
    with open(filename) as f:
       for line in f.readlines():
           if line.startswith('#'):
               continue
           values = line.split()
           if not values:
               continue
           if values[0] == 'v':
               V.append([float(x) for x in values[1:4]])
           elif values[0] == 'vt':
               T.append([float(x) for x in values[1:3]])
           elif values[0] == 'vn':
               N.append([float(x) for x in values[1:4]])
           elif values[0] == 'f' :
               Vi.append([int(indices.split('/')[0]) for indices in values[1:]])
               try:
                   Ti.append([int(indices.split('/')[1]) for indices in values[1:]])
               except:
                   pass
               try:
                   Ni.append([int(indices.split('/')[2]) for indices in values[1:]])
               except:
                   pass

    return fit(np.array(V)), np.array(Vi)-1

def normalize(V):
    """ Normalize V """

    return V/(1e-16+np.sqrt((np.array(V)**2).sum(axis=-1)))[..., np.newaxis]


def clamp(V, vmin=0, vmax=1):
    """ Clamp V between vmin and vmax """

    return np.minimum(np.maximum(V,vmin),vmax)


def viewport(x, y, w, h, d, dtype=np.float32, transpose=False):
    """ Viewport matrix

    Args:

        x (int):
            X origin (pixels) of the viewport (lower left)

        y (int):
            Y origin (pixels) of the viewport (lower left)

        h (int):
            Height (pixels) of the viewport

        w (int):
            Width (pixels) of the viewport

        d (float):
            Depth of the viewport.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Viewport matrix
    """

    M = np.array([[w/2, 0, 0, x+w/2],
                  [0, h/2, 0, y+h/2],
                  [0, 0, d/2,   d/2],
                  [0, 0, 0,       1]], dtype=dtype)
    if transpose:
        return np.transpose(M).view(mat4)
    else:
        return M.view(mat4)



def frustum(left, right, bottom, top, znear, zfar, dtype=np.float32, transpose=False):
    r"""View frustum matrix

    Args:

        left (float):
            Left coordinate of the field of view.

        right (float):
            Right coordinate of the field of view.

        bottom (float):
            Bottom coordinate of the field of view.

        top (float):
            Top coordinate of the field of view.

        znear (float):
            Near coordinate of the field of view.

        zfar (float):
            Far coordinate of the field of view.

        dtype (numpy dtype):
            dtype of the resulting array

        transpose (boolean):
            Whether to transpose result


    Returns:

        (mat4): View frustum matrix
    """

    M = np.zeros((4, 4), dtype=dtype)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0

    if transpose:
        return np.transpose(M).view(mat4)
    else:
        return M.view(mat4)


def perspective(fovy, aspect, znear, zfar, dtype=np.float32, transpose=False):
    """ Perspective projection matrix

    Args:

        fovy (float):
            The field of view along the y axis.

        aspect (float):
            Aspect ratio of the view.

        znear (float):
            Near coordinate of the field of view.

        zfar (float):
            Far coordinate of the field of view.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Perspective projection matrix
    """

    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar, dtype, transpose)


def ortho(left, right, bottom, top, znear, zfar, dtype=np.float32, transpose=False):
    """Create orthographic projection matrix

    Args:

        left (float):
            Left coordinate of the field of view.

        right (float):
            Right coordinate of the field of view.

        bottom (float):
            Bottom coordinate of the field of view.

        top (float):
            Top coordinate of the field of view.

        znear (float):
            Near coordinate of the field of view.

        zfar (float):
            Far coordinate of the field of view.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (boolean):
            Whether to transpose result

    Returns:

        (mat4): Orthographic projection matrix
    """

    M = np.zeros((4, 4), dtype=dtype)
    M[0, 0] = +2.0 / (right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 2] = -(right + left) / float(right - left)
    M[1, 3] = -(top + bottom) / float(top - bottom)
    M[2, 3] = -(zfar + znear) / float(zfar - znear)

    if transpose:
        return np.transpose(M).view(mat4)
    else:
        return M.view(mat4)

def lookat(eye=(0,0,4.5), center=(0,0,0), up=(0,0,1), dtype=np.float32, transpose=False):
    """
    Creates a viewing matrix derived from an eye point, a reference
    point indicating the center of the scene, and an up vector.

    Args:

        eye (vec3):
            Eye point

        center (vec3):
            Reference point

        up (vec3):
            Up vector

        dtype (np.dtype):
            dtype of the resulting array

        transpose (boolean):
            Whether to transpose result

    Returns:

        (mat4): View matrix
    """

    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)

    Z = normalize(eye - center)
    Y = up
    X = normalize(np.cross(Y, Z))
    Y = normalize(np.cross(Z, X))
    return np.array([
        [X[0], X[1], X[2], -np.dot(X, eye)],
        [Y[0], Y[1], Y[2], -np.dot(Y, eye)],
        [Z[0], Z[1], Z[2], -np.dot(Z, eye)],
        [0, 0, 0, 1]], dtype=dtype).view(mat4)


def scale(scale, dtype=np.float32, transpose=False):
    """Non-uniform scaling along the x, y, and z axes

    Args:

        scale (vec3):
            Scaling vector

        dtype (np dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Scaling matrix
    """

    x,y,z = np.array(scale)
    S = np.array([[x, 0, 0, 0],
                  [0, y, 0, 0],
                  [0, 0, z, 0],
                  [0, 0, 0, 1]], dtype=dtype)

    if transpose:
        return np.transpose(S).view(mat4)
    else:
        return S.view(mat4)


def fit(vertices):
    """ Fit vertices to the normalized cube

    Args:

        vertices (np.array): Vertices to fit

    Returns:

        (np.ndarray): vertices contained in the normalize cube
    """

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    # return 2*(vertices-vmin) / max(vmax-vmin)-1
    V = 2*(vertices - Vmin) / max(Vmax-Vmin) - 1
    return  V - (V.min(axis=0) + V.max(axis=0))/2


def translate(translate, dtype=np.float32, transpose=False):
    """
    Translation by a given vector

    Args:

        translate (vec3):
            Translation vector.

        dtype (np dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Translation matrix
    """

    x, y, z = np.array(translate)
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=dtype)

    if transpose:
        return np.transpose(T).view(mat4)
    else:
        return T.view(mat4)


def center(vertices):
    """ Center vertices around the origin.

    Args:

        vertices (np.array): Vertices to center

    Returns:

        (np.ndarray): vertices centered
    """

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return V - (vmax+vmin)/2


def xrotate(theta=0, dtype=np.float32, transpose=False):
    """Rotation about the X axis

    Args:

        theta (float):
            Specifies the angle of rotation, in degrees.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Rotation matrix
    """

    t = np.radians(theta)
    c, s = np.cos(t), np.sin(t)
    R =  np.array([[1, 0,  0, 0],
                   [0, c, -s, 0],
                   [0, s,  c, 0],
                   [0, 0,  0, 1]], dtype=dtype)

    if transpose:
        return np.transpose(R).view(mat4)
    else:
        return R.view(mat4)


def yrotate(theta=0, dtype=np.float32, transpose=False):
    """Rotation about the Y axis

    Args:

        theta (float):
            Specifies the angle of rotation, in degrees.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Rotation matrix
    """

    t = np.radians(theta)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[ c, 0, s, 0],
                  [ 0, 1, 0, 0],
                  [-s, 0, c, 0],
                  [ 0, 0, 0, 1]], dtype=dtype)

    if transpose:
        return np.transpose(R).view(mat4)
    else:
        return R.view(mat4)


def zrotate(theta=0, dtype=np.float32, transpose=False):
    """Rotation about the Z axis

    Args:

        theta (float):
            Specifies the angle of rotation, in degrees.

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Rotation matrix
    """

    t = np.radians(theta)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[ c, -s, 0, 0],
                  [ s,  c, 0, 0],
                  [ 0,  0, 1, 0],
                  [ 0,  0, 0, 1]], dtype=dtype)

    if transpose:
        return np.transpose(R).view(mat4)
    else:
        return R.view(mat4)


def rotate(theta, axis, dtype=np.float32, transpose=False):
    """Rotation about an arbitrary X axis

    Args:

        theta (float):
            Specifies the angle of rotation, in degrees.

        axis (vec3):
            Axis of rotation

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Rotation matrix
    """

    t = np.radians(theta)

    axis = normalize(np.array(axis))
    a = np.cos(t/2)
    b, c, d = -axis*np.sin(t/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    R =  np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
                   [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
                   [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
                   [0,0,0,1]], dtype=dtype)

    if transpose:
        return np.transpose(R).view(mat4)
    else:
        return R.view(mat4)



def align(U, V, dtype=np.float32, transpose=False):
    """
    Return the rotation matrix that aligns U to V

    Args:

        U (vec[234]):
            First vector

        U (vec[234]):
            Second vector

        dtype (np.dtype):
            dtype of the resulting array

        transpose (bool):
            Whether to transpose result

    Returns:

        (mat4): Rotation matrix
    """

    a, b = normalize(U), normalize(V)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    K = np.array([[   0, -v[2], v[1]],
                  [ v[2],   0, -v[0]],
                  [-v[1], v[0],   0]])
    R = np.zeros((4,4), dtype=dtype)
    R[:3, :3] = np.eye(3) + K + K@K * ((1 - c) / (s ** 2))
    R[3, 3] = 1

    if transpose:
        return np.transpose(R).view(mat4)
    else:
        return R.view(mat4)


def frontback(T):
    """
    Sort front and back facing triangles

    Parameters:
    -----------
    T : (n,3) array
       Triangles to sort

    Returns:
    --------
    front and back facing triangles as (n1,3) and (n2,3) arrays (n1+n2=n)
    """
    Z = (T[:,1,0]-T[:,0,0])*(T[:,1,1]+T[:,0,1]) + \
        (T[:,2,0]-T[:,1,0])*(T[:,2,1]+T[:,1,1]) + \
        (T[:,0,0]-T[:,2,0])*(T[:,0,1]+T[:,2,1])
    return Z < 0, Z >= 0


def camera(xrotation=25, yrotation=45, zoom=1, mode="perspective"):
    xrotation = min(max(xrotation, 0), 90)
    yrotation = min(max(yrotation, 0), 90)
    zoom = max(0.1, zoom)
    model = scale(zoom,zoom,zoom) @ xrotate(xrotation) @ yrotate(yrotation)
    view  = translate(0, 0, -4.5)
    if mode == "ortho":
        proj  = ortho(-1, +1, -1, +1, 1, 100)
    else:
        proj  = perspective(25, 1, 1, 100)
    return proj @ view  @ model
