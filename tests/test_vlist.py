# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import pytest
import numpy as np
from glm import vlist

def test_creation_1():

    V = vlist([np.ones(1), np.ones(2), np.ones(3)])
    
    assert(len(V) == 3)
    assert(np.array_equal(V.itemsize, [1,2,3]))
    assert(V.size == 6)

def test_creation_2():
    
    V = vlist(np.ones(6), [1,2,3])
    assert(len(V) == 3)
    assert(np.array_equal(V.itemsize, [1,2,3]))
    assert(V.size == 6)

def test_creation_3():
    
    V = vlist(np.ones(6), 2)
    assert(len(V) == 3)
    assert(np.array_equal(V.itemsize, [2,2,2]))
    assert(V.size == 6)

def test_creation_4():
    
    V = vlist(np.ones((10,2)), 2)
    assert(len(V) == 5)
    assert(V[0].shape == (2,2))

def test_creation_5():
    
    V = vlist([[0], [1,2], [3,4,5], [6,7,8,9]])
    assert(len(V) == 4)
    assert(np.array_equal(V.itemsize, [1,2,3,4]))
        
def test_setitem_1():
    
    V = vlist(np.zeros(6), 2)
    V[1] = 1
    assert(np.array_equal(V[1], [1,1]))

def test_setitem_2():
    
    V = vlist(np.zeros(10), 2)
    V[3:] = 1
    assert(np.array_equal(V[3:], [1,1,1,1]))

def test_setitem_3():
    
    V = vlist(np.zeros((10,2)), 2)
    V[0] = 1
    assert(np.array_equal(V[0], [[1,1],[1,1]]))
