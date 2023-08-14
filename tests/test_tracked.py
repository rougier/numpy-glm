import pytest
import numpy as np
from glm import ndarray
from glm import vlist
from glm import mat2, mat3, mat4
from glm import vec2, vec3, vec4

class Tracker:
    def __init__(self, shape, dtype):
        self._array = np.empty(shape, dtype=dtype)
    def set_data(self, offset, bytes):
        V = self._array.view(np.ubyte).ravel()
        V[offset:offset+len(bytes)] = np.frombuffer(bytes, dtype=np.ubyte)
    def __array__(self):
        return self._array

@pytest.fixture(autouse=True)
def run_around_tests():
    ndarray.tracked.__tracker_class__ = Tracker
    yield
    ndarray.tracked.__tracker_class__ = None


def test_creation():
    Z1 = ndarray.tracked(10)
    assert( isinstance(Z1._tracker, Tracker) )
    assert( np.asarray(Z1._tracker).shape == (10,))

    Z2 = Z1[1:-1]
    assert( isinstance(Z2._tracker, Tracker) )
    assert( Z1._tracker == Z2._tracker )

    Z = ndarray.tracked((10,10))
    assert( isinstance(Z._tracker, Tracker) )
    assert( np.asarray(Z._tracker).shape == (10,10))

def test_subclass():
    assert(isinstance(vec2(), ndarray.tracked))
    assert(isinstance(vec3(), ndarray.tracked))
    assert(isinstance(vec4(), ndarray.tracked))
    assert(isinstance(mat2(), ndarray.tracked))
    assert(isinstance(mat3(), ndarray.tracked))
    assert(isinstance(mat4(), ndarray.tracked))
    
def test_1d_tracking():
    Z = ndarray.tracked(10)
    Z[...] = 0
    Z[0] = 1
    Z[-1] = 2
    Z[3:8] = 3
    assert( np.array_equal(Z, np.asarray(Z._tracker)) )

def test_2d_tracking():
    Z = ndarray.tracked((3,3))
    Z[...] = 0
    Z[:,0] = 1
    Z[0,:] = 2
    Z[1:2, 1:2] = 3
    assert( np.array_equal(Z, np.asarray(Z._tracker)) )

def test_structured_tracking():
    Z = ndarray.tracked(10, dtype = [("x", float), ("y", float)])
    Z["x"] = 0
    Z["y"] = 1
    assert( np.array_equal(Z, np.asarray(Z._tracker)) )

def test_vec_tracking():
    Z = vec4(10)
    Z[...] = 0
    print(Z.shape, Z.dtype, Z._tracker._array.shape)
    assert( np.array_equal(Z, np.asarray(Z._tracker)) )

    Z = vec4(10)
    Z[...] = 0
    Z[0].z = 1
    assert( np.array_equal(Z, np.asarray(Z._tracker)) )

    
def test_fancy_tracking():
    Z = ndarray.tracked(10)
    with pytest.raises(NotImplementedError):
        Z[[0,1,2]] = 0
    
def test_partial_tracking():
    Z1 = np.zeros(10)
    Z2 = Z1[1:-1].view(ndarray.tracked)
    Z2[...] = 0
    Z2[0] = 1
    Z2[-1] = 2
    Z2[3:8] = 3
    assert( np.array_equal(Z2, np.asarray(Z2._tracker)) )

def test_view():
    Z = np.zeros(10).view(ndarray.tracked)
    assert( isinstance(Z._tracker, Tracker) )
    
    Z1 = np.zeros(10)
    Z2 = Z1[1:-1].view(ndarray.tracked)
    assert( isinstance(Z2._tracker, Tracker) )
    assert( np.asarray(Z2._tracker).shape == (8,))
