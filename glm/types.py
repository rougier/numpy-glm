# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
"""
Vector and matrices creation functions that return numpy arrays with swizzle capabilities.
"""

import numpy as np
from . tracked_array import tracked_array
from . swizzle_array import swizzle_array

ctypes  = [np.int8, np.uint8,
           np.int16, np.uint16, np.float16,
           np.int32, np.uint32, np.float32,
           np.int64, np.uint64, np.float64]

__all__ = [
    "bvec2", "ivec2", "uvec2", "hvec2", "vec2", "dvec2", "to_vec2", "as_vec2",
    "bvec3", "ivec3", "uvec3", "hvec3", "vec3", "dvec3", "to_vec3", "as_vec3",
    "bvec4", "ivec4", "uvec4", "hvec4", "vec4", "dvec4", "to_vec4", "as_vec4",
    "vec2_t", "vec3_t", "vec4_t",
    "bmat2", "imat2", "umat2", "hmat2", "mat2", "dmat2",
    "bmat3", "imat3", "umat3", "hmat3", "mat3", "dmat3",
    "bmat4", "imat4", "umat4", "hmat4", "mat4", "dmat4",
    "mat2_t", "mat3_t", "mat4_t"
]



def scalar_t(ctype):
    """ Returns a scalar numpy dtype with base type `ctype` """
    
    return np.dtype(ctype)

def vec2_t(ctype):
    """ vec2 dtype with base type `ctype` """
    
    return np.dtype((ctype, 2))

def vec3_t(ctype):
    """ vec3 dtype with base type `ctype` """
    
    return np.dtype((ctype, 3))

def vec4_t(ctype):
    """ vec4 dtype with base type `ctype` """
        
    return np.dtype((ctype, 4))

def mat2_t(ctype):
    """ mat2 dtype with base type `ctype`. """
        
    return np.dtype((ctype, (2,2)))

def mat3_t(ctype):
    """ mat3 dtype with base type `ctype`. """
        
    return np.dtype((ctype, (3,3)))

def mat4_t(ctype):
    """ mat4 dtype with base type `ctype`. """
    
    return np.dtype((ctype, (4,4)))

class scalar_array(tracked_array):
    """Array of scalars (tracked)"""
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, scalar_t(ctype),
                               buffer, offset, strides, order)

class vec2_array(swizzle_array):
    """2-components vectors"""
    
    swizzle = "xy", "ra"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec2_t(ctype),
                               buffer, offset, strides, order)

def as_vec2(other, dtype=None):
    """ View other as a vec2 with given dtype """
        
    V = np.asanyarray(other).reshape(-1,2)
    if dtype is None:
        return V.view(vec2_array)
    else:
        return V.astype(dtype).view(vec2_array)

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
        _V = vec2_array(V.size//2, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = vec2_array(len(V), dtype)
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
    return vec2_array(count, ctype=np.ubyte)

def ivec2(count=1):
    """2-components vectors of signed integers (32 bits) """
    return vec2_array(count, ctype=np.int32)

def uvec2(count=1):
    """2-components vectors of unsigned integers (32 bits) """
    return vec2_array(count, ctype=np.uint32)

def hvec2(count=1):
    """2-components vectors of half precision floats (16 bits) """
    return vec2_array(count, ctype=np.float16)

def vec2(count=1):
    """2-components vectors of single precision floats (32 bits) """
    return vec2_array(count, ctype=np.float32)

def dvec2(count=1):
    """2-components vectors of double precision floats (64 bits) """
    return vec2_array(count, ctype=np.float64)


class vec3_array(swizzle_array):
    """3-components vectors."""
        
    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec3_t(ctype),
                              buffer, offset, strides, order)

    
def as_vec3(other, dtype=None):
    """ View other as a vec3 with given dtype """    
    
    V = np.asanyarray(other).reshape(-1,3)
    if dtype is None:
        return V.view(vec3_array)
    else:
        return V.astype(dtype).view(vec3_array)

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
        _V = vec3_array(V.size//3, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = vec3_array(len(V), dtype)
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
    return vec3_array(count, ctype=np.uint8)

def ivec3(count=1):
    """3-components vectors of signed integers (32 bits) """
    return vec3_array(count, ctype=np.int32)

def uvec3(count=1):
    """3-components vectors of unsigned integers (32 bits) """
    return vec3_array(count, ctype=np.uint32)

def hvec3(count=1):
    """3-components vectors of half precision floats (16 bits) """
    return vec3_array(count, ctype=np.float16)

def vec3(count=1):
    """3-components vectors of single precision floats (32 bits) """
    return vec3_array(count, ctype=np.float32)

def dvec3(count=1):
    """3-components vectors of double precision floats (64 bits) """
    return vec3_array(count, ctype=np.float64)



class vec4_array(swizzle_array):
    """4-components vectors"""
    
    swizzle = "xyzw", "rgba"

    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec4_t(ctype),
                               buffer, offset, strides, order)

def as_vec4(other, dtype=None):
    """ View other as a vec4 with given dtype """    
        
    V = np.asanyarray(other).reshape(-1,4)
    if dtype is None:
        return V.view(vec4_array)
    else:
        return V.astype(dtype).view(vec4_array)

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
        _V = vec4_array(V.size//4, dtype)
        _V.ravel()[...] = V
        return _V        
    elif len(V.shape) == 2:
        _V = vec4_array(len(V), dtype)
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
    return vec4_array(count, ctype=np.uint8)

def ivec4(count=1):
    """4-components vectors of signed integers (32 bits) """
    return vec4_array(count, ctype=np.int32)

def uvec4(count=1):
    """4-components vectors of unsigned integers (32 bits) """
    return vec4_array(count, ctype=np.uint32)

def hvec4(count=1):
    """4-components vectors of half precision floats (16 bits) """
    return vec4_array(count, ctype=np.float16)

def vec4(count=1):
    """4-components vectors of single precision floats (32 bits) """
    return vec4_array(count, ctype=np.float32)

def dvec4(count=1):
    """4-components vectors of double precision floats (64 bits) """
    return vec4_array(count, ctype=np.float64)


class mat2_array(swizzle_array):
    """2x2 matrices"""

    swizzle = "xy", "ra"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat2_t(ctype),
                               buffer, offset, strides, order)

def bmat2(count=1):
    """2x2 matrices of booleans (8 bits)"""
    return mat2_array(count, ctype=np.uint8)

def imat2(count=1):
    """2x2 matrices of integers (32 bits)"""
    return mat2_array(count, ctype=np.int32)

def umat2(count=1):
    """2x2 matrices of unsigned integers (32 bits)"""
    return mat2_array(count, ctype=np.uint32)

def hmat2(count=1):
    """2x2 matrices of half precision floats (16 bits)"""
    return mat2_array(count, ctype=np.float16)

def mat2(count=1):
    """2x2 matrices of double precision floats (32 bits)"""
    return mat2_array(count, ctype=np.float32)

def dmat2(count=1):
    """2x2 matrices of double precision floats (64 bits)"""
    return mat2_array(count, ctype=np.float64)


class mat3_array(swizzle_array):
    """3x3 matrices"""

    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat3_t(ctype),
                               buffer, offset, strides, order)

def bmat3(count=1):
    """3x3 matrices of booleans (8 bits)"""
    return mat3_array(count, ctype=np.uint8)

def imat3(count=1):
    """3x3 matrices of integers (32 bits)"""
    return mat3_array(count, ctype=np.int32)

def umat3(count=1):
    """3x3 matrices of unsigned integers (32 bits)"""
    return mat3_array(count, ctype=np.uint32)

def hmat3(count=1):
    """3x3 matrices of half precision floats (16 bits)"""
    return mat3_array(count, ctype=np.float16)

def mat3(count=1):
    """3x3 matrices of single precision floats (32 bits)"""
    return mat3_array(count, ctype=np.float32)

def dmat3(count=1):
    """3x3 matrices of double precision floats (64 bits)"""
    return mat3_array(count, ctype=np.float64)


class mat4_array(swizzle_array):
    """4x4 matrices"""

    swizzle = "xyzw", "rgba"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat4_t(ctype),
                               buffer, offset, strides, order)

def bmat4(count=1):
    """4x4 matrices of booleans (8 bits)"""
    return mat4_array(count, ctype=np.uint8)

def imat4(count=1):
    """4x4 matrices of integers (32 bits)"""
    return mat4_array(count, ctype=np.int32)

def umat4(count=1):
    """4x4 matrices of unsigned integers (32 bits)"""
    return mat4_array(count, ctype=np.uint32)

def hmat4(count=1):
    """4x4 matrices of half precision floats (16 bits)"""
    return mat4_array(count, ctype=np.float16)

def mat4(count=1):
    """4x4 matrices of single precision floats (32 bits)"""
    return mat4_array(count, ctype=np.float32)

def dmat4(count=1):
    """4x4 matrices of double precision floats (64 bits)"""
    return mat4_array(count, ctype=np.float64)



