# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
""" Matrices creation functions """

import numpy as np
from . import ndarray

def mat2_t(dtype):
    """ mat2 dtype """
    return np.dtype((dtype, (2,2)))

def mat3_t(dtype):
    """ mat3 dtype """
    return np.dtype((dtype, (3,3)))

def mat4_t(dtype):
    """ mat4 dtype """
    return np.dtype((dtype, (4,4)))

def bmat2(count=1):
    """2x2 matrices of booleans (8 bits)"""
    return ndarray.mat2(count, dtype=np.uint8)

def imat2(count=1):
    """2x2 matrices of integers (32 bits)"""
    return ndarray.mat2(count, dtype=np.int32)

def umat2(count=1):
    """2x2 matrices of unsigned integers (32 bits)"""
    return ndarray.mat2(count, dtype=np.uint32)

def hmat2(count=1):
    """2x2 matrices of half precision floats (16 bits)"""
    return ndarray.mat2(count, dtype=np.float16)

def mat2(count=1):
    """2x2 matrices of double precision floats (32 bits)"""
    return ndarray.mat2(count, dtype=np.float32)

def dmat2(count=1):
    """2x2 matrices of double precision floats (64 bits)"""
    return ndarray.mat2(count, dtype=np.float64)

def bmat3(count=1):
    """3x3 matrices of booleans (8 bits)"""
    return ndarray.mat3(count, dtype=np.uint8)

def imat3(count=1):
    """3x3 matrices of integers (32 bits)"""
    return ndarray.mat3(count, dtype=np.int32)

def umat3(count=1):
    """3x3 matrices of unsigned integers (32 bits)"""
    return ndarray.mat3(count, dtype=np.uint32)

def hmat3(count=1):
    """3x3 matrices of half precision floats (16 bits)"""
    return ndarray.mat3(count, dtype=np.float16)

def mat3(count=1):
    """3x3 matrices of single precision floats (32 bits)"""
    return ndarray.mat3(count, dtype=np.float32)

def dmat3(count=1):
    """3x3 matrices of double precision floats (64 bits)"""
    return ndarray.mat3(count, dtype=np.float64)

def bmat4(count=1):
    """4x4 matrices of booleans (8 bits)"""
    return ndarray.mat4(count, dtype=np.uint8)

def imat4(count=1):
    """4x4 matrices of integers (32 bits)"""
    return ndarray.mat4(count, dtype=np.int32)

def umat4(count=1):
    """4x4 matrices of unsigned integers (32 bits)"""
    return ndarray.mat4(count, dtype=np.uint32)

def hmat4(count=1):
    """4x4 matrices of half precision floats (16 bits)"""
    return ndarray.mat4(count, dtype=np.float16)

def mat4(count=1):
    """4x4 matrices of single precision floats (32 bits)"""
    return ndarray.mat4(count, dtype=np.float32)

def dmat4(count=1):
    """4x4 matrices of double precision floats (64 bits)"""
    return ndarray.mat4(count, dtype=np.float64)



