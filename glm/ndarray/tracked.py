# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
""" Memory aware array class """
import numpy as np

class tracked(np.ndarray):
    """
    A tracked array keeps track of the smallest contiguous block
    of modified memory and can signals a tracker of any change through
    the tracker `set_data` method. A minimal tracker class can thus be
    written as:

    ```python
    class Tracker:
        def __init__(self, shape, dtype):
            self._array = np.empty(shape, dtype=dtype)
        def set_data(self, offset, bytes):
            V = self._array.view(np.ubyte).ravel()
            V[offset:offset+len(bytes)] = np.frombuffer(bytes, dtype=np.ubyte)
    ```

    It is the responsability of the user to set the __tracker_class__
    class attribute with the relevant tracker class, prior to the
    creation of any tracked array. If this class is not None, the
    tracker is created at the time of array creation.

    Maintaining a copy of an array as shown in the example above is not very
    interesting. However, you can modify the tracker to mirror a CPU array
    in GPU, by uploading the new data to the GPU in the `set_data` method.
    """
    
    __tracker_class__ = None
    
    def __new__(cls, *args, **kwargs):
        obj = np.ndarray.__new__(cls, *args, **kwargs)
        if cls.__tracker_class__ is not None:
            obj._tracker = cls.__tracker_class__(obj.shape, obj.dtype)
        return obj

    def __array_finalize__(self, obj):

        if not isinstance(obj, tracked):
            self.__class__.__init__(self)
        self._extents = 0, self.size*self.itemsize
        self._dirty = self._extents
        self._tracker = getattr(obj, '_tracker', None)
        if self._tracker is None and self.__tracker_class__ is not None:
            self._tracker = self.__tracker_class__(self.shape, self.dtype)

    def clear(self):
        """ Clear dirty region"""

        if isinstance(self.base, tracked):
            self.base._dirty = None
        elif self._dirty:
            self._dirty = None

    @property
    def dirty(self):
        """ Dirty region as (start, stop) in bytes """

        if isinstance(self.base, tracked):
            return self.base.dirty
        elif self._dirty:
            return self._dirty
        return None

    def _update(self, start, stop):
        """ Update dirty region """
        
        if isinstance(self.base, tracked):
            self.base._update(start, stop)
        else:
            if not hasattr(self, "_dirty") or self._dirty is None:
                self._dirty = start, stop
            else:
                start = min(self._dirty[0], start)
                stop = max(self._dirty[1], stop)
                self._dirty = start, stop
                
    def _compute_extents(self, Z):
        """Compute extents (start, stop) in bytes in the base array"""

        if Z.base is not None:
            base = Z.base.__array_interface__['data'][0]
            view = Z.__array_interface__['data'][0]
            offset = view - base
            shape = np.array(Z.shape) - 1
            strides = Z.strides[-len(shape):]
            size = (shape*strides).sum() + Z.itemsize
            return offset, offset+size
        return 0, Z.size*Z.itemsize

    def __getitem__(self, key):
        Z = np.ndarray.__getitem__(self, key)
        if not hasattr(Z, 'shape') or Z.shape == ():
            return Z        
        Z._extents = self._compute_extents(Z)
        return Z

    def __setitem__(self, key, value):
        Z = np.ndarray.__getitem__(self, key)
        
        if Z.shape == ():
            # This test for the case of [...,index] notation. Since we
            # know the result is a scalar, we can safely remove the
            # ellipsis component (that should be the first item).
            if (isinstance(key, tuple)):
                key = tuple([k for k in key if k is not Ellipsis])
            key = tuple(np.mod(np.array(key), self.shape))
            offset = np.ravel_multi_index(key, self.shape, mode='wrap')*self.itemsize
            self._update(offset, offset+self.itemsize)
                    
        # Test for fancy indexing
        elif (Z.base is not self and (isinstance(key, list) or
               (hasattr(key, '__iter__') and
                any(isinstance(k, (list,np.ndarray)) for k in key)))):
            raise NotImplementedError("Fancy indexing not supported")
        else:
            Z._extents = self._compute_extents(Z)            
            self._update(Z._extents[0], Z._extents[1])
        np.ndarray.__setitem__(self, key, value)

        if self._tracker:
            data = self.view(np.ubyte).ravel()
            start, stop = self._dirty
            self._tracker.set_data(start, data[start:stop].tobytes())
            self.clear()

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop,  value):
        self.__setitem__(slice(int(start), int(stop)), value)

    def __iadd__(self, other):
        self._update(self._extents[0], self._extents[1])
        return np.ndarray.__iadd__(self, other)

    def __isub__(self, other):
        self._update(self._extents[0], self._extents[1])
        return np.ndarray.__isub__(self, other)

    def __imul__(self, other):
        self._update(self._extents[0], self._extents[1])
        return np.ndarray.__imul__(self, other)

    def __idiv__(self, other):
        self._update(self._extents[0], self._extents[1])
        return np.ndarray.__idiv__(self, other)


