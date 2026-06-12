# Use this command to build
# python setup.py build_ext --inplace

import os
import sys

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_namespace_packages, setup

on_rtd = os.environ.get("READTHEDOCS") == "True"

extra_compile_args = []
extra_link_args = []

if not on_rtd:
    if sys.platform == "win32":
        extra_compile_args += ["/O2", "/openmp"]
    else:
        extra_compile_args += ["-O3", "-fopenmp"]
        extra_link_args += ["-fopenmp"]

ext_modules = []

if not on_rtd:
    ext_modules = [
        Pybind11Extension(
            "axionbloch.blochsimulation",
            [
                "blochSimulation_ext/bloch_wrapper.cpp",
                "blochSimulation_ext/bloch.cpp",
            ],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            cxx_std=17,
        )
    ]

setup(
    name="axionbloch",
    version="0.2.1",
    packages=find_namespace_packages(include=["axionbloch", "axionbloch.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
