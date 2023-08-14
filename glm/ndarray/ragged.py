# -----------------------------------------------------------------------------
# GL Mathematics for Numpy
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
"""
A ragged array is a strongly typed list whose type can be anything that can be
interpreted as a numpy data type.

Example
-------

```python
>>> L = ragged_array( [[0], [1,2], [3,4,5], [6,7,8,9]] )
>>> print L
[ [0] [1 2] [3 4 5] [6 7 8 9] ]
>>> print L.data
[0 1 2 3 4 5 6 7 8 9]
```

You can add several items at once by specifying common or individual size: a
single scalar means all items are the same size while a list of sizes is used
to specify individual item sizes.

Example
-------

```python
>>> L = ragged_array( np.arange(10), [3,3,4])
>>> print L
[ [0 1 2] [3 4 5] [6 7 8 9] ]
>>> print L.data
[0 1 2 3 4 5 6 7 8 9]
```
"""
import numpy as np

class ragged:
    """
    An ragged_array is a strongly typed list whose type can be anything that can
    be interpreted as a numpy data type.
    """

    def __init__(self, data=None, itemsize=None):
        """ Create a new buffer using given data and sizes or dtype

        Parameters
        ----------

        data : array_like
            An array, any object exposing the array interface, an object
            whose __array__ method returns an array, or any (nested) sequence.

        itemsize:  int or 1-D array
            If `itemsize is an integer, N, the array will be divided
            into elements of size N. If such partition is not possible,
            an error is raised.

            If `itemsize` is 1-D array, the array will be divided into
            elements whose succesive sizes will be picked from itemsize.
            If the sum of itemsize values is different from array size,
            an error is raised.
        """

        if isinstance(data, (list, tuple)):
            itemsize = np.array([len(l) for l in data], dtype=int)
            if isinstance(data[0], np.ndarray):
                data = np.concatenate(data, axis=0).view(data[0].__class__)
            else:
                data = np.concatenate(data, axis=0)
            
        data = np.asanyarray(data)
        if itemsize is None:
            itemsize = len(data)

        if isinstance(itemsize, int):
            if (len(data) % itemsize) != 0:
                raise ValueError("Cannot partition data as requested")
            count = len(data) // itemsize
            itemsize = np.ones(count, dtype=int) * (len(data) // count)
        else:
            if np.sum(itemsize) != len(data):
                raise ValueError("Cannot partition data as requested")
            count = len(itemsize)

        self._data = data
        self._size = len(data)
        self._count = count
        self._items = np.zeros((count, 2), int)
        C = np.cumsum(itemsize)
        self._items[1:, 0] += C[:-1]
        self._items[0:, 1] += C
            

    def __getitem__(self, key):
        """ x.__getitem__(y) <==> x[y] """

        if isinstance(key, (list, tuple)):
            indices = []
            for k in key:
                start,stop = self._items[k]
                indices.extend(list(range(start,stop)))
            return self._data[indices]
        else:
            indices = self._items[key]
            start, stop = indices.min(), indices.max()
            return self._data[start:stop]

    def __setitem__(self, key, value):
        """ x.__setitem__(y, v) <==> x[y] = v"""

        if isinstance(key, (list, tuple)):
            indices = []
            for k in key:
                start,stop = self._items[k]
                indices.extend(list(range(start,stop)))
            self._data[indices] = value
        else:
            indices = self._items[key]
            start, stop = indices.min(), indices.max()
            self._data[start:stop] = value


    @property
    def data(self):
        """ The array's elements, in memory. """
        return self._data

    @property
    def size(self):
        """ Number of base elements, in memory. """
        return self._size

    @property
    def itemsize(self):
        """ Individual item sizes """
        return self._items[:self._count, 1] - self._items[:self._count, 0]

    @property
    def dtype(self):
        """ Describes the format of the elements in the buffer. """
        return self._data.dtype

    @property
    def shape(self):
        """ Describes the format of the elements in the buffer. """
        return self._data.shape

    def __len__(self):
        """ x.__len__() <==> len(x) """
        return self._count

    def __str__(self):
        s = '[ '
        for item in self:
            s += str(item) + ' '
        s += ']'
        return s
    
    def __array__(self):
        return self.data
    

