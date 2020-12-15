import numpy as np
from scipy.linalg import solve_banded

import randomvars._utils as utils

# %% Retyping
def _y_from_xp(x, p, metric):
    """Compute y-grid from xp-grid

    Compute y-grid which together with input x-grid is dual to input xp-grid.
    Duality is defined in terms of distance between CDF functions of
    corresponding distributions: xy-grid will represent Cont-distribution
    closest to Disc-distribution with input xp-grid. Distance is measured in
    terms of input `metric` (one of "L1" or "L2").

    In other words, output y-grid will be from the closest Cont-distribution to
    Disc-distribution with xp-grid among all Cont-distributions with the same
    x-grid.

    Output y-grid is computed so that CDF values of input Disc-distribution
    inside its constant intervals are "centers" of CDF values of output
    Cont-distribution:
      - For L1 metric CDF value of Disc-distribution is median.
      - For L2 metric CDF value of Disc-distribution is mean.

    **Note**: there is no L_infinity metric because it doesn't represent
    one-to-one conversion. This is despite the fact that it can be easily used
    when computing p-grid from xy-grid (CDF value of Disc-distribution would be
    midpoint with metric coefficient equal to 0.5). It can be implemented in a
    hacky approximate way (for example, by using metric coefficient 0.5+1e-6),
    but currently this doesn't seem like a good idea.
    """
    coeff = metric_coeffs[metric]

    dx = np.diff(x)
    dx_lead = np.concatenate([dx, [0]])
    dx_lag = np.concatenate([[0], dx])

    banded_matrix = 0.5 * np.array(
        [dx_lag * (1 - coeff), (dx_lag + dx_lead) * coeff, dx_lead * (1 - coeff)]
    )

    return solve_banded(l_and_u=(1, 1), ab=banded_matrix, b=p)


def _p_from_xy(x, y, metric):
    """Compute p-grid from xy-grid

    Compute p-grid which together with input x-grid is dual to input xy-grid.
    Duality is defined in terms of distance between CDF functions of
    corresponding distributions: xp-grid will represent Disc-distribution
    closest to Cont-distribution with input xy-grid. Distance is measured in
    terms of input `metric` (one of "L1" or "L2").

    In other words, output p-grid will be from the closest Disc-distribution to
    Cont-distribution with xy-grid among all Disc-distributions with the same
    x-grid.

    Output p-grid is computed so that CDF values of output Disc-distribution
    inside its constant intervals are "centers" of CDF values of input
    Cont-distribution:
      - For L1 metric CDF value of Disc-distribution is median.
      - For L2 metric CDF value of Disc-distribution is mean.

    **Note**: there is no L_infinity metric because it doesn't represent
    one-to-one conversion. This is despite the fact that it can be easily
    implemented (CDF value of Disc-distribution would be midpoint with metric
    coefficient equal to 0.5). It can be implemented for computing y-grid from
    xp-grid in a hacky approximate way (for example, by using metric
    coefficient 0.5+1e-6), but currently this doesn't seem like a good idea.
    """
    coeff = metric_coeffs[metric]

    cump = utils._trapez_integral_cum(x, y)
    dx = np.diff(x)

    disc_cump = cump[:-1] + 0.5 * dx * (coeff * y[:-1] + (1 - coeff) * y[1:])

    return np.diff(disc_cump, prepend=0, append=1)


metric_coeffs = {"L1": 0.75, "L2": 2 / 3}
