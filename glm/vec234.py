# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
"""
Vector creation and conversion functions
"""

import numpy as np
from . import ndarray

def vec2_t(dtype):
    """ vec2 dtype """
    return np.dtype((dtype, 2))

def vec3_t(dtype):
    """ vec3 dtype """
    return np.dtype((dtype, 3))

def vec4_t(dtype):
    """ vec4 dtype """
    return np.dtype((dtype, 4))

def as_vec2(other, dtype=None):
    """ View other as a vec2 with given dtype """
        
    V = np.asanyarray(other).reshape(-1,2)
    if dtype is None:
        return V.view(ndarray.vec2)
    else:
        return V.astype(dtype).view(ndarray.vec2)

def to_vec2(other, dtype=None):
    """ Convert other to a vec2:

    * v    -> vec2 : v[::2] -> v[i],v[i+1]
    * vec2 -> vec2 : x,y     -> x,y
    * vec3 -> vec2 : x,y,z   -> x,y
    * vec4 -> vec2 : x,y,z,w -> x/w,y/w
    """

    V = np.array(other)
    dtype = dtype if dtype is not None else V.dtype

    if len(V.shape) == 1 and V.size % 2 == 0:
        _V = ndarray.vec2(V.size//2, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = ndarray.vec2(len(V), dtype)
        _V[...] = V[:,:2]
        if V.shape[1] == 2:
            return _V
        elif V.shape[1] == 3:
            return _V
        elif V.shape[1] == 4:
            _V /= V[:,3, np.newaxis]
            return _V
    
    raise TypeError("Cannot convert %s to vec2" % other) 
    
def bvec2(count=1):
    """2-components vectors of unsigned bytes (8 bits) """
    return ndarray.vec2(count, dtype=np.ubyte)

def ivec2(count=1):
    """2-components vectors of signed integers (32 bits) """
    return ndarray.vec2(count, dtype=np.int32)

def uvec2(count=1):
    """2-components vectors of unsigned integers (32 bits) """
    return ndarray.vec2(count, dtype=np.uint32)

def hvec2(count=1):
    """2-components vectors of half precision floats (16 bits) """
    return ndarray.vec2(count, dtype=np.float16)

def vec2(count=1):
    """2-components vectors of single precision floats (32 bits) """
    return ndarray.vec2(count, dtype=np.float32)

def dvec2(count=1):
    """2-components vectors of double precision floats (64 bits) """
    return ndarray.vec2(count, dtype=np.float64)

    
def as_vec3(other, dtype=None):
    """ View other as a vec3 with given dtype """    
    
    V = np.asanyarray(other).reshape(-1,3)
    if dtype is None:
        return V.view(ndarray.vec3)
    else:
        return V.astype(dtype).view(ndarray.vec3)

def to_vec3(other, dtype=None):
    """ Convert other to a vec3

    * v    -> vec3 : v[::3]  -> v[i],v[i+1],v[i+2]
    * vec2 -> vec3 : x,y     -> x,y,0
    * vec3 -> vec3 : x,y,z   -> x,y,z
    * vec4 -> vec3 : x,y,z,w -> x/w,y/w,z/w
    """

    V = np.array(other)
    dtype = dtype if dtype is not None else V.dtype

    if len(V.shape) == 1 and V.size % 3 == 0:
        _V = ndarray.vec3(V.size//3, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = ndarray.vec3(len(V), dtype)
        if V.shape[1] == 2:
            _V[:,:2] = V
            _V[:, 2] = 0
            return _V
        elif V.shape[1] == 3:
            _V[...] = V
            return _V        
        elif V.shape[1] == 4:
            _V[...] = V[:,:3] / V[:,3, np.newaxis]
            return _V
    
    raise TypeError("Cannot convert %s to vec3" % other) 

def bvec3(count=1):
    """3-components vectors of booleans (8 bits) """
    return ndarray.vec3(count, dtype=np.uint8)

def ivec3(count=1):
    """3-components vectors of signed integers (32 bits) """
    return ndarray.vec3(count, dtype=np.int32)

def uvec3(count=1):
    """3-components vectors of unsigned integers (32 bits) """
    return ndarray.vec3(count, dtype=np.uint32)

def hvec3(count=1):
    """3-components vectors of half precision floats (16 bits) """
    return ndarray.vec3(count, dtype=np.float16)

def vec3(count=1):
    """3-components vectors of single precision floats (32 bits) """
    return ndarray.vec3(count, dtype=np.float32)

def dvec3(count=1):
    """3-components vectors of double precision floats (64 bits) """
    return ndarray.vec3(count, dtype=np.float64)


def as_vec4(other, dtype=None):
    """ View other as a vec4 with given dtype """    
        
    V = np.asanyarray(other).reshape(-1,4)
    if dtype is None:
        return V.view(ndarray.vec4)
    else:
        return V.astype(dtype).view(ndarray.vec4)

def to_vec4(other, dtype=None):
    """ Convert other to a vec4

    * v    -> vec4 : v[::4] -> v[i],v[i+1],v[i+2],v[i+3]
    * vec2 -> vec4 : x,y     -> x,y,0,1
    * vec3 -> vec4 : x,y,z   -> x,y,z,1
    * vec4 -> vec4 : x,y,z,w -> x,y,z,w
    """

    V = np.array(other)
    dtype = dtype if dtype is not None else V.dtype

    if len(V.shape) == 1 and V.size % 4 == 0:
        _V = ndarray.vec4(V.size//4, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = ndarray.vec4(len(V), dtype)
        if V.shape[1] == 2:
            _V[:,:2] = V
            _V[:, 2] = 0
            _V[:, 3] = 1
            return _V
        elif V.shape[1] == 3:
            _V[:,:3] = V
            _V[:, 3] = 1
            return _V        
        elif V.shape[1] == 4:
            _V[...] = V
            return _V
    
    raise TypeError("Cannot convert %s to vec3" % other) 

def bvec4(count=1):
    """4-components vectors of booleans (8 bits) """
    return ndarray.vec4(count, dtype=np.uint8)

def ivec4(count=1):
    """4-components vectors of signed integers (32 bits) """
    return ndarray.vec4(count, dtype=np.int32)

def uvec4(count=1):
    """4-components vectors of unsigned integers (32 bits) """
    return ndarray.vec4(count, dtype=np.uint32)

def hvec4(count=1):
    """4-components vectors of half precision floats (16 bits) """
    return ndarray.vec4(count, dtype=np.float16)

def vec4(count=1):
    """4-components vectors of single precision floats (32 bits) """
    return ndarray.vec4(count, dtype=np.float32)

def dvec4(count=1):
    """4-components vectors of double precision floats (64 bits) """
    return ndarray.vec4(count, dtype=np.float64)

