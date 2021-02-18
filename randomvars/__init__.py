""" randomvars: Work with Random Variables
"""

from .options import config, _uses_options
from ._random import Rand
from ._boolean import Bool
from ._continuous import Cont
from ._discrete import Disc
from ._mixture import Mixt
from .downgrid_maxtol import downgrid_maxtol

# Update `config` with registered options
config.__doc__ = config.__doc__ + _uses_options.get_option_desc()
del _uses_options
