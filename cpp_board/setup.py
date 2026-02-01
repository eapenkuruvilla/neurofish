"""
setup.py - Build and install chess_cpp Python extension

This script builds the C++ chess library with pybind11 bindings.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            parallel_level = os.cpu_count() or 1
            build_args += [f"-j{parallel_level}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )


setup(
    name="chess_cpp",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fast C++ chess library with Python bindings",
    long_description=open("../README.md").read() if os.path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("chess_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["pybind11>=2.6.0"],
)
