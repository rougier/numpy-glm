
# Introduction

GLM offers a set of objects and functions to ease the design of 3D applications while taking advantage of the numpy library. The main objects are vectors (`vec2`, `vec3`, `vec4`), matrices (`mat2`, `mat3`, `mat4`) and vectorized lists (`vlist`).  Vectors and matrices possess several variants depending on the base type and are memory tracked. This means it is possible to know anytime the smallest contiguous block of memory that has changed. This can be used to maintain a GPU copy up-to-date.
