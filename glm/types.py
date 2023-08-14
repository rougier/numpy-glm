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

__all__ = [ "bvec2", "ivec2", "uvec2", "hvec2", "vec2", "dvec2",
            "bvec3", "ivec3", "uvec3", "hvec3", "vec3", "dvec3",
            "bvec4", "ivec4", "uvec4", "hvec4", "vec4", "dvec4",
            "vec2_t", "vec3_t", "vec4_t",
            "bmat2", "imat2", "umat2", "hmat2", "mat2", "dmat2",
            "bmat3", "imat3", "umat3", "hmat3", "mat3", "dmat3",
            "bmat4", "imat4", "umat4", "hmat4", "mat4", "dmat4",
            "mat2_t", "mat3_t", "mat4_t" ]


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

class _scalar(tracked_array):
    """Array of scalars (tracked)"""
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, scalar_t(ctype),
                               buffer, offset, strides, order)

class _vec2(swizzle_array):
    """2-components vectors"""
    
    swizzle = "xy", "ra"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec2_t(ctype),
                               buffer, offset, strides, order)

def bvec2(count=1):
    """2-components vectors of booleans (8 bits) """
    return _vec2(count, ctype=np.uint8)

def ivec2(count=1):
    """2-components vectors of signed integers (32 bits) """
    return _vec2(count, ctype=np.int32)

def uvec2(count=1):
    """2-components vectors of unsigned integers (32 bits) """
    return _vec2(count, ctype=np.uint32)

def hvec2(count=1):
    """2-components vectors of half precision floats (16 bits) """
    return _vec2(count, ctype=np.float16)

def vec2(count=1):
    """2-components vectors of single precision floats (32 bits) """
    return _vec2(count, ctype=np.float32)

def dvec2(count=1):
    """2-components vectors of double precision floats (64 bits) """
    return _vec2(count, ctype=np.float64)


class _vec3(swizzle_array):
    """3-components vectors."""
        
    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec3_t(ctype),
                              buffer, offset, strides, order)

def bvec3(count=1):
    """3-components vectors of booleans (8 bits) """
    return _vec3(count, ctype=np.uint8)

def ivec3(count=1):
    """3-components vectors of signed integers (32 bits) """
    return _vec3(count, ctype=np.int32)

def uvec3(count=1):
    """3-components vectors of unsigned integers (32 bits) """
    return _vec3(count, ctype=np.uint32)

def hvec3(count=1):
    """3-components vectors of half precision floats (16 bits) """
    return _vec3(count, ctype=np.float16)

def vec3(count=1):
    """3-components vectors of single precision floats (32 bits) """

    if is_vec4(count):
        return vec4_to_vec3(count)
    return _vec3(count, ctype=np.float32)


def dvec3(count=1):
    """3-components vectors of double precision floats (64 bits) """
    return _vec3(count, ctype=np.float64)




class _vec4(swizzle_array):
    """4-components vectors"""
    
    swizzle = "xyzw", "rgba"

    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, vec4_t(ctype),
                               buffer, offset, strides, order)

def bvec4(count=1):
    """4-components vectors of booleans (8 bits) """
    return _vec4(count, ctype=np.uint8)

def ivec4(count=1):
    """4-components vectors of signed integers (32 bits) """
    return _vec4(count, ctype=np.int32)

def uvec4(count=1):
    """4-components vectors of unsigned integers (32 bits) """
    return _vec4(count, ctype=np.uint32)

def hvec4(count=1):
    """4-components vectors of half precision floats (16 bits) """
    return _vec4(count, ctype=np.float16)

def vec4(count=1):
    """4-components vectors of single precision floats (32 bits) """

    if is_vec3(count):
        return vec3_to_vec4(count)    
    return _vec4(count, ctype=np.float32)

def dvec4(count=1):
    """4-components vectors of double precision floats (64 bits) """
    return _vec4(count, ctype=np.float64)


class _mat2(swizzle_array):
    """2x2 matrices"""

    swizzle = "xy", "ra"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat2_t(ctype),
                               buffer, offset, strides, order)

def bmat2(count=1):
    """2x2 matrices of booleans (8 bits)"""
    return _mat2(count, ctype=np.uint8)

def imat2(count=1):
    """2x2 matrices of integers (32 bits)"""
    return _mat2(count, ctype=np.int32)

def umat2(count=1):
    """2x2 matrices of unsigned integers (32 bits)"""
    return _mat2(count, ctype=np.uint32)

def hmat2(count=1):
    """2x2 matrices of half precision floats (16 bits)"""
    return _mat2(count, ctype=np.float16)

def mat2(count=1):
    """2x2 matrices of double precision floats (32 bits)"""
    return _mat2(count, ctype=np.float32)

def dmat2(count=1):
    """2x2 matrices of double precision floats (64 bits)"""
    return _mat2(count, ctype=np.float64)


class _mat3(swizzle_array):
    """3x3 matrices"""

    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat3_t(ctype),
                               buffer, offset, strides, order)

def bmat3(count=1):
    """3x3 matrices of booleans (8 bits)"""
    return _mat3(count, ctype=np.uint8)

def imat3(count=1):
    """3x3 matrices of integers (32 bits)"""
    return _mat3(count, ctype=np.int32)

def umat3(count=1):
    """3x3 matrices of unsigned integers (32 bits)"""
    return _mat3(count, ctype=np.uint32)

def hmat3(count=1):
    """3x3 matrices of half precision floats (16 bits)"""
    return _mat3(count, ctype=np.float16)

def mat3(count=1):
    """3x3 matrices of single precision floats (32 bits)"""
    return _mat3(count, ctype=np.float32)

def dmat3(count=1):
    """3x3 matrices of double precision floats (64 bits)"""
    return _mat3(count, ctype=np.float64)


class _mat4(swizzle_array):
    """4x4 matrices"""

    swizzle = "xyzw", "rgba"
    
    def __new__(subtype, count=1, ctype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        return super().__new__(subtype, count, mat4_t(ctype),
                               buffer, offset, strides, order)

def bmat4(count=1):
    """4x4 matrices of booleans (8 bits)"""
    return _mat4(count, ctype=np.uint8)

def imat4(count=1):
    """4x4 matrices of integers (32 bits)"""
    return _mat4(count, ctype=np.int32)

def umat4(count=1):
    """4x4 matrices of unsigned integers (32 bits)"""
    return _mat4(count, ctype=np.uint32)

def hmat4(count=1):
    """4x4 matrices of half precision floats (16 bits)"""
    return _mat4(count, ctype=np.float16)

def mat4(count=1):
    """4x4 matrices of single precision floats (32 bits)"""
    return _mat4(count, ctype=np.float32)

def dmat4(count=1):
    """4x4 matrices of double precision floats (64 bits)"""
    return _mat4(count, ctype=np.float64)



def is_vec(V, n):
    return (isinstance(V, np.ndarray)
            and len(V.shape) == 2
            and V.shape[1] == n
            and V.dtype in [np.float32, np.float64])

def is_vec2(V): return is_vec(V, 2)
def is_vec3(V): return is_vec(V, 3)
def is_vec4(V): return is_vec(V, 4)

def vec2_to_vec3(V2):
    """ vec2 to vec3 (z=0) """
    V3 = _vec3(len(V2), V2.dtype)
    V3[:, :2] = V2
    V3[:, 2] = 0
    return V3

def vec2_to_vec4(V2):
    """ vec2 to vec4 (z=0, w=1) """
    V4 = _vec4(len(V2), V2.dtype)
    V4[:, :2] = V2
    V4[:, 2:] = 0, 1
    return V4

def vec3_to_vec2(V3):
    """ vec3 to vec2 """
    V2 = _vec2(len(V3), V3.dtype)
    V2[...] = V3[...,:2]
    return V2

def vec3_to_vec4(V3):
    """ vec3 to vec4 (w=1) """
    V4 = _vec4(len(V3), V3.dtype)
    V4[:, :3] = V3
    V4[:, 3] = 1
    return V4

def vec4_to_vec3(V4):
    """ vec4 to vec4 (x/w, y/w, z/w) """
    V3 = _vec3(len(V4), V4.dtype)
    V3[...] = V4[:,:3] / V4[:, 3, np.newaxis]
    return V3

def vec4_to_vec2(V4):
    """ vec4 to vec4 (x/w, y/w) """
    V2 = _vec2(len(V4), V4.dtype)
    V2[...] = V4[:,:2] / V4[:, 3, np.newaxis]
    return V2


