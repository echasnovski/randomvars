from scipy.stats.kde import gaussian_kde

from randomvars._utils import default_discrete_estimator


__all__ = [
    "get_option",
    "set_option",
    "reset_option",
    "option_context",
    "OptionError",
    "default_discrete_estimator",
]

_default_options = {
    "density_estimator": gaussian_kde,
    "density_mincoverage": 0.9999,
    "discrete_estimator": default_discrete_estimator,
    "n_grid": 1001,
    "tail_prob": 1e-6,
    "integr_tol": 1e-4,
    "tolerance": (0.0, 1e-8),
}
_options = _default_options.copy()

_options_list = """
- density_estimator : callable, default scipy.stats.kde.gaussian_kde. Function
  which takes sample as input and returns one of:
    - Callable object for density estimate (takes points as input and returns
      density values).
    - Object of class `Cont` or `rv_frozen` (`rv_continuous` with all
      hyperparameters defined).
  **Notes**:
    - Theoretical integral of density over whole real line should be 1.
    - Output density callable should be vectorized: allow numpy array as input
      and return numpy array with the same length.
    - There is worse performance if output density callable has discontinuity.
- density_mincoverage : float, default 0.9999. Minimum value of integral within
  output of density range estimation.
- discrete_estimator : callable, default
  randomvars.options.default_discrete_estimator. Function which takes sample as
  input and returns one of:
    - Tuple with two elements representing `x` and `prob` of discrete distribution.
    - Object of class `Disc` or `rv_frozen` (`rv_discrete` with all
      hyperparameters defined).
- n_grid : int, default 1001. Number of points in initial xy-grids when
  creating object of class `Cont`.
- tail_prob : float, default 1e-6. Probability value of distribution tail that
  might be cutoff in order to get finite support.
- integr_tol : float, default 1e-4. Integral tolerance for maximum tolerance
  downgridding. Used to ensure that difference of total integrals between input
  and downgridded xy-grids is less than `integr_tol`.
- tolerance : tuple with two elements, default (0.0, 1e-8). Tuple with
  relative and absolute tolerance to be used as `rtol` and `atol` arguments in
  `numpy.isclose()` and similar functions. **Note** that default values are
  different than in `numpy.isclose()`.
"""


def get_option(opt):
    try:
        return _options[opt]
    except KeyError:
        raise OptionError(f"There is no option '{opt}'.")


get_option.__doc__ = f"""
Get package option

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.

Raises
------
OptionError : if no such option exists.
"""


def set_option(opt, val):
    # Ensure that `opt` option can be accessed, raising relevant error
    # otherwise
    get_option(opt)

    # Set option
    _options[opt] = val


set_option.__doc__ = f"""
Set package option

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.
val : any
    Option value.

Raises
------
OptionError : if no such option exists.
"""


def reset_option(opt):
    # Ensure that `opt` option can be accessed, raising relevant error
    # otherwise
    get_option(opt)

    # Reset option
    set_option(opt, _default_options[opt])


reset_option.__doc__ = f"""
Reset package option to default

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.

Raises
------
OptionError : if no such option exists.
"""


class option_context:
    def __init__(self, opt_dict):
        self.opt_dict = opt_dict

    def __enter__(self):
        self.undo = {opt: get_option(opt) for opt in self.opt_dict}

        for opt, val in self.opt_dict.items():
            set_option(opt, val)

    def __exit__(self, *args):
        if self.undo:
            for opt, val in self.undo.items():
                set_option(opt, val)


option_context.__doc__ = f"""
Context manager to temporarily set options in the `with` statement context.

List of available options:
{_options_list}

Parameters
----------
opt_dict : dict
    Dictionary with option names as keys and option values as values.
"""


class OptionError(KeyError):
    """
    Exception describing error in interaction with package options.
    """
