import numpy as np
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, splrep, splev
import scipy.stats.distributions as distrs
import matplotlib.pyplot as plt

from randomvars import Disc, Cont
from randomvars._continuous import _xy_from_cdf_spline
import randomvars._utils as utils
import randomvars._utilsgrid as utilsgrid


def apply_fun(f, xp1, xp2=None):
    if xp2 is None:
        x, p = xp1
        x_res = f(x)
        p_res = p
    else:
        x1, p1 = xp1
        x2, p2 = xp2
        x_res = np.concatenate([f(x, x2) for x in x1])
        p_res = np.concatenate([p * p2 for p in p1])

    return aggregate_xp(x_res, p_res)


def aggregate_xp(x, p, tol=1e-12):
    x_raw, inds = np.unique(x, return_inverse=True)
    p_raw = np.bincount(inds, weights=p)

    x_is_good = np.concatenate([[True], np.ediff1d(x_raw) > tol])
    agg_inds = np.cumsum(x_is_good) - 1
    x_res = x_raw[x_is_good]
    p_res = np.bincount(agg_inds, weights=p_raw)
    return x_res, p_res


def y_from_xp_2(x, p):
    cump_inner = np.cumsum(p)[:-1]
    cump_mid = 0.5 * (cump_inner[:-1] + cump_inner[1:])
    cump = np.concatenate(([0], cump_mid, [1]))
    cdf_spline = LSQUnivariateSpline(x=x, y=cump, t=x[1:-1], k=2)
    return _xy_from_cdf_spline(cdf_spline)


def modify_xy(x, y):
    y = np.clip(y, 0, None)
    y = y / np.trapz(y, x)
    return x, y


# %% Function output application
# rv_input = distrs.norm()
# rv_ref = distrs.norm(scale=np.sqrt(2))
# rv1 = Cont.from_rv(distrs.norm())
# xy1 = rv1.x, rv1.y
# xp1 = xy1[0], utilsgrid._p_from_xy(*xy1, metric="L2")
# xp2 = xp1

rv_input = distrs.norm()
rv_ref = distrs.norm(scale=np.sqrt(2))
x1 = np.linspace(-3, 3, 101)
y1 = rv_input.pdf(x1)
y1 = y1 / np.trapz(y1, x1)
p1 = utilsgrid._p_from_xy(x1, y1, "L2")
xp1 = x1, p1
xp2 = xp1

# xy1 = np.linspace(0, 1, 101), np.repeat(1, 101)
# xp1 = xy1[0], utilsgrid._p_from_xy(*xy1, metric="L2")
# xy2 = xy1
# xp2 = xp1

xp_fun = apply_fun(f=lambda x, y: x + y, xp1=xp1, xp2=xp2)
xy_fun = xp_fun[0], utilsgrid._y_from_xp(*xp_fun, "L2")

x_grid = np.linspace(-3, 3, 1001)
plt.plot(*xy_fun, label="reconstructed")
plt.plot(x_grid, rv_ref.pdf(x_grid), label="reference")
plt.legend()
plt.show()

plt.plot(x_grid, Cont(*modify_xy(*xy_fun)).cdf(x_grid), label="reconstructed")
plt.plot(x_grid, rv_ref.cdf(x_grid), label="reference")
plt.legend()
plt.show()
# CONCLUSION: function application through conversion works great if xy-grids
# are equidistant in terms of x. Otherwise output is really bad.


# %% From sample reconstruction
rv = distrs.norm()
smpl = rv.rvs(size=100, random_state=101)

rv_disc = Disc.from_sample(smpl)
x, p = rv_disc.x, rv_disc.p
y = utilsgrid._y_from_xp(x, p, "L2")

x_equi = np.linspace(rv_disc.a, rv_disc.b, 101)
p_equi = np.diff(rv_disc.cdf(x_equi), prepend=0)
y_equi = utilsgrid._y_from_xp(x_equi, p_equi, "L2")

x_grid = np.linspace(rv.ppf(1e-6), rv.ppf(1 - 1e-6), 1001)
plt.plot(x, y)
plt.plot(x_equi, y_equi)
plt.plot(x_grid, rv.pdf(x_grid))
plt.show()
