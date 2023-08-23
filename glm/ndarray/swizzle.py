# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
import numpy as np
from . tracked import tracked

class swizzle(tracked):
    """A swizzle array allows to access the last dimension of an
    array using a virtual attribute whose name is specified at the
    class level. Typical usage is:

    ```python
    class vec4(swizzle_array):
        swizzle = "xyzw", "rgba"

    v = vec4()
    v.xyzw = 1,2,3,4 # equivalent to v[[0,1,2,3]] = 1,2,3,4
    v.wzyx = 1,2,3,4 # equivalent to v[[3,2,1,0]] = 1,2,3,4
    ```
    """
    
    swizzle = None

    def __getattr__(self, key):
        for swizzle in self.swizzle:
            if set(key).issubset(set(swizzle)):
                return self[..., [swizzle.index(c) for c in key]]
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        """
        """
        
        for swizzle in self.swizzle:
            if set(key).issubset(set(swizzle)):
                value = np.asarray(value)
                shape = value.shape
                indices = [swizzle.index(c) for c in key]

                # Z.xy = 1
                if not len(shape):
                    for index in indices:
                        self[..., index] = value
                    break
                # Z.xy = np.ones(10)
                elif len(self) == value.size: # shape[-1] == 1:
                    for index in indices:
                        self[..., index] = np.squeeze(value)
                    break
                # Z.xy = np.ones(10), np.ones(10)
                elif len(key) == len(value):
                    for src_index, tgt_index in enumerate(indices):
                        self[..., tgt_index] = value[src_index]
                    break
                # Z.xy = np.ones(10,2)
                elif shape[-1] == len(key):
                    for tgt_index, src_index in enumerate(indices):
                        if self[...,src_index].size == value[...,tgt_index].size:
                            self[...,src_index] = value[...,tgt_index].reshape(self[...,src_index].shape)
                        else:
                            self[...,src_index] = value[...,tgt_index]
                    break
                else:
                    raise IndexError
        else:
            super().__setattr__(key, value)
