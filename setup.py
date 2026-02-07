# Use this command to build
# python setup.py build_ext --inplace

# from setuptools import setup
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# import pybind11
# import sys

# extra_compile_args = []
# extra_link_args = []

# if sys.platform == "win32":
#     extra_compile_args += ["/O2", "/arch:AVX2", "/openmp"]
# else:
#     extra_compile_args += ["-O3", "-march=native", "-fopenmp"]
#     extra_link_args += ["-fopenmp"]

# ext_modules = [
#     Pybind11Extension(
#         "blochSimulation_c_ext.blochSimulation_c",
#         [
#             "blochSimulation_c_ext/bloch_wrapper.cpp",  # pybind11 wrapper
#             "blochSimulation_c_ext/bloch.cpp",  # native C++ code
#         ],
#         include_dirs=[pybind11.get_include(), "blochSimulation_c_ext"],
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#     )
# ]

# setup(
#     name="blochSimulation_c",
#     version="0.0.1",
#     description="Spin Dynamics in python",
#     author="Yuzhe Zhang",
#     author_email="yuhzhang@uni-mainz.de",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": build_ext},
#     zip_safe=False,
# )

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys
import os

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
            "axionbloch._bloch_ext",
            [
                "blochsimulation_ext/bloch_wrapper.cpp",
                "blochsimulation_ext/bloch.cpp",
            ],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            cxx_std=17,
        )
    ]

setup(
    name="axionbloch",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
