# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
""" Matrices creation functions """

import numpy as np
from . import ndarray

def bmat2(count = None):
    """2x2 matrices of booleans (8 bits)

    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.ubyte
    """
    return ndarray.mat2(count, dtype=np.uint8)

def imat2(count = None):
    """2x2 matrices of integers (32 bits)
    
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.int8
    """
    return ndarray.mat2(count, dtype=np.int32)

def umat2(count = None):
    """2x2 matrices of unsigned integers (32 bits)
        
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.uint8
    """
    return ndarray.mat2(count, dtype=np.uint32)

def hmat2(count = None):
    """2x2 matrices of half precision floats (16 bits)
        
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.float16
    """
    return ndarray.mat2(count, dtype=np.float16)

def mat2(count = None):
    """2x2 matrices of double precision floats (32 bits)
        
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.float32
    """
    return ndarray.mat2(count, dtype=np.float32)

def dmat2(count = None):
    """2x2 matrices of double precision floats (64 bits)
        
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 2, 2) shaped numpy array with dtype np.float64
    """
    return ndarray.mat2(count, dtype=np.float64)

def bmat3(count = None):
    """3x3 matrices of booleans (8 bits)
        
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.uint8
    """
    return ndarray.mat3(count, dtype=np.uint8)

def imat3(count = None):
    """3x3 matrices of integers (32 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.int32
    """
    return ndarray.mat3(count, dtype=np.int32)

def umat3(count = None):
    """3x3 matrices of unsigned integers (32 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.uint32
    """
    return ndarray.mat3(count, dtype=np.uint32)

def hmat3(count = None):
    """3x3 matrices of half precision floats (16 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.float16
    """
    return ndarray.mat3(count, dtype=np.float16)

def mat3(count = None):
    """3x3 matrices of single precision floats (32 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.float32
    """
    return ndarray.mat3(count, dtype=np.float32)

def dmat3(count = None):
    """3x3 matrices of double precision floats (64 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 3, 3) shaped numpy array with dtype np.float64
    """
    return ndarray.mat3(count, dtype=np.float64)

def bmat4(count = None):
    """4x4 matrices of booleans (8 bits)
            
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.uint8
    """
    return ndarray.mat4(count, dtype=np.uint8)

def imat4(count = None):
    """4x4 matrices of integers (32 bits)
                
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.int32
    """
    return ndarray.mat4(count, dtype=np.int32)

def umat4(count = None):
    """4x4 matrices of unsigned integers (32 bits)
                
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.uint32
    """
    return ndarray.mat4(count, dtype=np.uint32)

def hmat4(count = None):
    """4x4 matrices of half precision floats (16 bits)
                
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.float16
    """
    return ndarray.mat4(count, dtype=np.float16)

def mat4(count = None):
    """4x4 matrices of single precision floats (32 bits)
                
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.float32
    """
    return ndarray.mat4(count, dtype=np.float32)

def dmat4(count = None):
    """4x4 matrices of double precision floats (64 bits)
                
    Args:

        count (int): Number of vectors to create

    Returns:

        (np.ndarray): (count, 4, 4) shaped numpy array with dtype np.float64
    """
    return ndarray.mat4(count, dtype=np.float64)



