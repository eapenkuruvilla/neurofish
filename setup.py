"""
Setup script for building Cython extensions.

Build with:
    python setup.py build_ext --inplace

Or using cythonize directly:
    cythonize -i nn_ops_fast.pyx
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "nn_ops_fast",
        sources=["nn_ops_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="nn_ops_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
