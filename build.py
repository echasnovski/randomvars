"""Build Cython modules"""
from Cython.Build import cythonize
import numpy


extensions = cythonize(
    "randomvars/downgrid_maxtol.pyx",
    include_path=[numpy.get_include()],
    language_level=3,
)


def build(setup_kwargs):
    """Build Cython modules"""
    setup_kwargs.update({"ext_modules": extensions})
