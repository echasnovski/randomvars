# Module 'downgrid_maxtol'

This directory contains code for Cython version of `downgrid_maxtol()` function.

To make it usable inside 'experiments' folder (via simple `from downgrid_maxtol import downgrid_maxtol`), run `./build.sh` from this directory. This will all build Cython code and make binary library ('downgrid_maxtol.*.so') available in 'experiments' folder.
