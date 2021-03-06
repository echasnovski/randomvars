import numpy as np

from randomvars import Cont, Disc, Mixt
import randomvars._utils as utils

# my_cont = Cont([0, 1], [1, 1])
# my_disc = Disc([0.5], [1])
# weight_cont = 0.5

# my_mixt = Mixt(my_cont, my_disc, weight_cont)

# rv = my_mixt
# h = 1e-12

# rv.cdf([0, 0.5 - h, 0.5, 1])

# # Case of discrete part inside zero density region of continuous part
my_cont = Cont([0, 1, 2, 3, 4, 5], [0, 0.5, 0, 0, 0.5, 0])
my_disc = Disc([2.125, 2.875], [0.25, 0.75])
weight_cont = 0.5
my_mixt = Mixt(my_cont, my_disc, weight_cont)
rv = my_mixt
h = 1e-12

q = rv.cdf([0, 1, 2.125 - h, 2.125, 2.5, 2.875 - h, 2.875, 4, 5])


def _compute_cump_tuple(rv):
    # Case of one discrete part
    if rv._missing_cont():
        disc_x = rv._disc.x
        cump = np.concatenate([0, rv._disc.cdf(disc_x)])
        x = np.concatenate([rv._disc.a, disc_x])
        ids = np.repeat("d", len(x) - 1)
        return cump, x, ids

    # Case of one continuous part
    if rv._missing_disc():
        cont_x = rv._cont.x
        cump = rv._cont.cdf(cont_x)
        x = cont_x
        ids = np.repeat("c", len(x) - 1)
        return cump, x, ids

    disc_x = rv._disc.x
    x = np.repeat(disc_x, 2)

    cump_right = rv.cdf(disc_x)
    cump_left = cump_right - rv._weight_disc * rv._disc.p
    ## Interchange elements of `cump_left` and `cump_right`
    cump = np.array([cump_left, cump_right], order="C").flatten(order="F")

    ids = np.tile(["d", "c"], len(disc_x))[:-1]

    # Add possibly missing intervals because of continuous part tails
    if cump[0] > 0:
        x = np.concatenate([[rv._cont.a], x])
        cump = np.concatenate([[0], cump])
        ids = np.concatenate([["c"], ids])
    if cump[-1] < 1:
        x = np.concatenate([x, [rv._cont.b]])
        cump = np.concatenate([cump, [1]])
        ids = np.concatenate([ids, ["c"]])

    # print(f"{x=}, {cump=}, {ids=}")
    return cump, x, ids


def mixt_ppf(rv, q):
    q = np.asarray(q, dtype="float64")
    res = np.zeros_like(q, dtype="float64")

    cump, x, ids = _compute_cump_tuple(rv)

    interval_inds = utils._searchsorted_wrap(cump, q, edge_inside=True) - 1

    # Process intervals from discrete part
    is_in_disc = ids[interval_inds] == "d"
    disc_inds = interval_inds[is_in_disc]
    res[is_in_disc] = x[disc_inds]

    # Process intervals from continuous part
    # `w_c * F_c(x) + w_d * F_d(x) = q`. It is known that `q` is such that it
    # lies inside interval (?possibly zero length?) created by continuous part.
    is_in_cont = ids[interval_inds] == "c"
    cont_inds = interval_inds[is_in_cont]
    disc_ppf = (
        cump[cont_inds] - rv._weight_cont * rv._cont.cdf(x[cont_inds])
    ) / rv._weight_disc
    q_mod = (q[is_in_cont] - rv._weight_disc * disc_ppf) / rv._weight_cont
    res[is_in_cont] = rv._cont.ppf(q_mod)

    return res
