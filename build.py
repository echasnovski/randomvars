"""Build Cython modules"""
from Cython.Build import cythonize
import numpy


extensions = cythonize("randomvars/downgrid_maxtol.pyx")


def build(setup_kwargs):
    """Build Cython modules"""
    setup_kwargs.update(
        {"ext_modules": extensions, "include_dirs": [numpy.get_include()]}
    )
