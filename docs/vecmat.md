# Vectors & Matrices

The generic notation for creating vectors and matrices is:

* `TvecN(n)` for vectors
* `TmatN(n)` for matrices,

with T in [`b`, `i` , `u`, `h`, `d`, `f`, Ø], N in *[2,3,4]* and *n* being the number of vectors or matrices to create.

| Symbol | Type             | Size    | Dtype         |
|--------|------------------|---------|---------------|
| `b`    | unsigned byte    | 8 bits  | `np.ubyte`    |
| `i`    | signed integer   | 32 bits | `np.int32`    |
| `u`    | unsigned integer | 32 bits | `np.uint32`   |
| `h`    | float            | 16 bits | `np.float16`  |
|  Ø     | float            | 32 bits | `np.float32`  |
| `h`    | float            | 64 bits | `np.float64`  |

* `TvecN(n)` returns a numpy array with shape *(n,N)* and dtype *T*.
* `TmatN(n)` returns a numpy array with shape *(n,N,N)* and dtype *T*.

**Example usage**

```python
import glm

# 10 x 3-components vectors of integers (32 bits)
V = glm.ivec3(10)

# 10 x 3-components vectors of float (32 bits)
V = glm.vec3(10)

# A 4x4 matrix of floats (32 bits)
M = glm.mat4()
```

The last components of vectors and matrices can be accessed using
named attributes (swizzling):

| Type                            | Index | Attribute  |
|---------------------------------|-------|------------|
| `Tvec[2,3,4]` / `Tmat[2,3,4]`   | 0     | `x` or `r` |
| `Tvec[2,3,4]` / `Tmat[2,3,4]`   | 1     | `y` or `g` |
| `Tvec[3,4]` / `Tmat[3,4]`       | 2     | `z` or `b` |
| `Tvec[4]` / `Tmat[4]`           | 3     | `w` or `a` |

**Example usage**

```python
import glm

V = glm.vec2(10)
V.xyz = 1,2,3 # or V[...,[0,1,2]] = 1,2,3
V.xxx = 1,2,3 # or V[...,[0,0,0]] = 1,2,3

V = glm.uvec4()
V.rgba = 0,0,0,255 # or V[...,[1,2,3,4]] = 0,0,0,255
```


