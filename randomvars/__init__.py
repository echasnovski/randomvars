""" randomvars: Work with Random Variables
"""

from .options import config, _uses_options, _option_desc
from ._random import Rand
from ._boolean import Bool
from ._continuous import Cont
from ._discrete import Disc
from ._mixture import Mixt
from .downgrid_maxtol import downgrid_maxtol

# Update `config` with registered options
option_desc = _uses_options.update_option_desc(_option_desc)
config.__doc__ = config.__doc__ + "\n".join(option_desc.values())
del option_desc
del _uses_options
del _option_desc
