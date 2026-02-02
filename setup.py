"""
Setup script for building optimized Cython extensions.

Build with:
    python setup.py build_ext --inplace

Compiler flags explained:
- -O3: Maximum optimization
- -ffast-math: Allow aggressive floating-point optimizations
- -march=native: Use all CPU instructions available (AVX2 on your system)
- -funroll-loops: Unroll loops for better pipelining
- -ftree-vectorize: Enable auto-vectorization (implicit with -O3)
- -fno-semantic-interposition: Better inlining for shared libs
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Aggressive optimization flags for GCC
extra_compile_args = [
    "-O3",
    "-ffast-math",
    "-march=native",
    "-funroll-loops",
    "-ftree-vectorize",
    "-fno-semantic-interposition",
    # Help auto-vectorizer
    "-fassociative-math",
    "-fno-signed-zeros",
    "-fno-trapping-math",
]

extensions = [
    Extension(
        "nn_ops_fast",
        sources=["nn_ops_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
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
            "nonecheck": False,
            "overflowcheck": False,
            # Enable this to see what's being optimized
            # "annotation_typing": True,
        },
    ),
)