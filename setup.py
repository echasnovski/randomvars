"""randomvars: Work with Random Variables
"""

from setuptools import setup

from Cython.Build import cythonize
import numpy

setup(
    name="randomvars",
    version="0.0.0.9000",
    description="Working with Random Variables",
    url="http://github.com/echasnovski/randomvars",
    author="Evgeni Chasnovski",
    author_email="evgeni.chasnovski@gmail.com",
    license="MIT",
    packages=["randomvars"],
    ext_modules=cythonize("randomvars/regrid_maxtol.pyx"),
    include_dirs=[numpy.get_include()],
)
