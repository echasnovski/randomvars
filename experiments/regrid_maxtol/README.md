# Module 'regrid_maxtol'

This directory contains code for Cython version of `regrid_maxtol()` function.

To make it usable inside 'experiments' folder (via simple `from regrid_maxtol import regrid_maxtol`), run `./build.sh` from this directory. This will all build Cython code and make binary library ('regrid_maxtol.*.so') available in 'experiments' folder.
