import pytest
import numpy as np
from glm import *

def test_dtypes():
    assert(ndarray.vec2_t(np.float32).base == np.float32)
    assert(ndarray.vec2_t(np.float32).shape == (2,))
    assert(ndarray.vec3_t(np.float32).base == np.float32)
    assert(ndarray.vec3_t(np.float32).shape == (3,))
    assert(ndarray.vec4_t(np.float32).base == np.float32)
    assert(ndarray.vec4_t(np.float32).shape == (4,))

def test_default_type():
    assert(vec2().dtype == np.float32)
    assert(vec3().dtype == np.float32)
    assert(vec4().dtype == np.float32)

def test_default_shape():
    assert(vec2().shape == (2,))
    assert(vec3().shape == (3,))
    assert(vec4().shape == (4,))

def test_dtype():
    assert(bvec2(1).dtype == np.ubyte)
    assert(bvec3(1).dtype == np.ubyte)
    assert(bvec4(1).dtype == np.ubyte)

def test_shape():
    assert(vec2(10).shape == (10,2))
    assert(vec3(10).shape == (10,3))
    assert(vec4(10).shape == (10,4))

    assert(vec2((3,3)).shape == (3,3,2))
    assert(vec3((3,3)).shape == (3,3,3))
    assert(vec4((3,3)).shape == (3,3,4))

def test_swap():
    Z = vec2(1)
    
    Z.xy = 1,2
    assert(np.array_equal(Z, [[1.,2.]]))
    
    Z.xy = Z.yx
    assert(np.array_equal(Z, [[2.,1.]]))
    
    Z.yx = Z.xy
    assert(np.array_equal(Z, [[1.,2.]]))
