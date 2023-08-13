# -----------------------------------------------------------------------------
# Numpy/GL Mathematics
# Copyright 2023 Nicolas P. Rougier - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
"""
GLM offers a set of objects and functions for 3D geometry inspired by the OpenGL API and the GLSL language.
"""

from . glm import *
from . types import *

from . ragged_array import ragged_array as vlist
from . tracked_array import tracked_array
from . swizzle_array import swizzle_array

from . trackball import Trackball
from . shapes import sphere, cube


