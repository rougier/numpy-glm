# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
""" Vector classes and base types """

import numpy as np
from . tracked import tracked
from . swizzle import swizzle

__all__ = [ "vec2",  "vec3", "vec4",  "vec2_t", "vec3_t", "vec4_t" ]

def scalar_t(dtype):
    """ scalar dtype """
    
    return np.dtype(dtype)

def vec2_t(dtype):
    """ vec2 dtype """
    
    return np.dtype((dtype, 2))

def vec3_t(dtype):
    """ vec3 dtype """
    
    return np.dtype((dtype, 3))

def vec4_t(dtype):
    """ vec4 dtype """
        
    return np.dtype((dtype, 4))

class scalar(tracked):
    """Array of scalars (tracked)"""
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, scalar_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, scalar_t(dtype),
                                buffer, offset, strides, order)

class vec2(swizzle):
    """2 components vectors"""
    
    swizzle = "xy", "ra"
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, vec2_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, vec2_t(dtype),
                            buffer, offset, strides, order)

class vec3(swizzle):
    """3 components vectors."""
        
    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, vec3_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, vec3_t(dtype),
                            buffer, offset, strides, order)

class vec4(swizzle):
    """4 components vectors"""
    
    swizzle = "xyzw", "rgba"

    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, vec4_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, vec4_t(dtype),
                            buffer, offset, strides, order)
