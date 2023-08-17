# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
""" Matrices classes """

import numpy as np
from . swizzle import swizzle

__all__ = [ "mat2", "mat3", "mat4", "mat2_t", "mat3_t", "mat4_t" ]

def mat2_t(dtype):
    """ mat2 dtype """
        
    return np.dtype((dtype, (2,2)))

def mat3_t(dtype):
    """ mat3 dtype """
        
    return np.dtype((dtype, (3,3)))

def mat4_t(dtype):
    """ mat4 dtype """
    
    return np.dtype((dtype, (4,4)))

class mat2(swizzle):
    """2x2 matrices"""

    swizzle = "xy", "ra"
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, mat2_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, mat2_t(dtype),
                               buffer, offset, strides, order)

class mat3(swizzle):
    """ 3x3 matrices """

    swizzle = "xyz", "rgb"
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, mat3_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, mat3_t(dtype),
                               buffer, offset, strides, order)

class mat4(swizzle):
    """4x4 matrices"""

    swizzle = "xyzw", "rgba"
    
    def __new__(subtype, count=None, dtype=np.float32, buffer=None,
                offset=0, strides=None, order=None, info=None):
        if count is None:
            return super().__new__(subtype, 1, mat4_t(dtype),
                                   buffer, offset, strides, order).squeeze()
        return super().__new__(subtype, count, mat4_t(dtype),
                               buffer, offset, strides, order)
