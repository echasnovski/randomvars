import numpy as np
import scipy.stats.distributions as distrs
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from randomvars import Cont, Disc
from randomvars._continuous import _xy_from_cdf_spline
import randomvars._utilsgrid as utilsgrid


def smooth_cont(rv, smooth=0.0):
    if smooth == 0:
        return rv

    x = rv.x
    q_x = rv.cdf(x)

    q_left = np.clip(q_x - smooth, 0, 1)
    x_left = rv.ppf(q_left)
    q_right = np.clip(q_x + smooth, 0, 1)
    x_right = rv.ppf(q_right)

    y_smoothed = (q_right - q_left) / (x_right - x_left)
    # y_smoothed = smooth / (x_right - x_left)
    y_smoothed = y_smoothed / np.trapz(y_smoothed, x)
    return Cont(x=x, y=y_smoothed)


def smooth_cont_cdf(rv, smooth=0.0):
    x = rv.x
    cdf_vals = rv.cdf(x)
    uniform_cdf_vals = (x - x[0]) / (x[-1] - x[0])
    cdf_smoothed_vals = (1 - smooth) * cdf_vals + smooth * uniform_cdf_vals
    cdf_spline = UnivariateSpline(x=x, y=cdf_smoothed_vals, k=2, s=0)
    density_spline = cdf_spline.derivative()
    y_smoothed = np.clip(density_spline(x), 0, None)
    return Cont(x=x, y=y_smoothed)


def smooth_cont_sample(rv, n_sample=1000):
    smpl = rv.rvs(size=n_sample, random_state=101)
    return Cont.from_sample(smpl)


def smooth_cont_kde(rv, smooth=None):
    x, a, b = rv.x, rv.a, rv.b
    rv_disc = rv.convert("Disc")
    kde = gaussian_kde(rv_disc.x, bw_method=smooth, weights=rv_disc.p)
    y_smoothed = kde(x) + kde(a - (x - a)) + kde(b + (b - x))
    return Cont(x=x, y=y_smoothed)


def smooth_cont_kde_2(rv, smooth=0.5):
    # `smooth` is betwen 0 (no smoothing) and 1 (total smoothing to uniform)
    x, y, a, b = rv.x, rv.y, rv.a, rv.b
    rv_disc = rv.convert("Disc")
    # Reference smoothing that deals with edge bias
    kde = gaussian_kde(rv_disc.x, bw_method=None, weights=rv_disc.p)
    y_kde = kde(x) + kde(a - (x - a)) + kde(b + (b - x))

    # Make 0.5 a natural middle point
    if smooth < 0.5:
        alpha = 2 * smooth
        y_smoothed = (1 - alpha) * y + alpha * y_kde
    else:
        alpha = 2 * smooth - 1
        # Extremely smoothed density is one-segment density computed via
        # fitting single-interval spline to CDF
        cdf_spline_extreme = UnivariateSpline(x=x, y=rv.cdf(x), k=2, s=np.inf)
        x_extreme, y_extreme = _xy_from_cdf_spline(cdf_spline_extreme)
        y_extreme_smooth = np.interp(x, x_extreme, y_extreme, left=0, right=0)
        y_smoothed = (1 - alpha) * y_kde + alpha * y_extreme_smooth

    return Cont(x=x, y=y_smoothed)


# rv_scipy = distrs.norm()
# # rv_scipy = distrs.expon()
# # rv_scipy = distrs.beta(a=4, b=1)
# # rv_scipy = distrs.uniform()
# smpl = rv_scipy.rvs(size=1000, random_state=101)
# rv_disc = Disc.from_sample(smpl)
# rv_cont = rv_disc.convert("Cont")
# rv = rv_cont
# x_grid = np.linspace(np.min(smpl), np.max(smpl), 1001)

rv_scipy = distrs.norm()
rv_cont = Cont.from_rv(rv_scipy)
x_grid = np.linspace(rv_cont.a, rv_cont.b, 1001)

for smooth in np.linspace(0, 1, 11):
    # rv_smoothed = smooth_cont(rv_cont, smooth=smooth)
    # rv_smoothed = smooth_cont_cdf(rv_cont, smooth=smooth)
    rv_smoothed = smooth_cont_kde_2(rv_cont, smooth=smooth)
    plt.plot(x_grid, rv_scipy.pdf(x_grid), label="True")
    plt.plot(rv_smoothed.x, rv_smoothed.y, label="Smoothed")
    # plt.plot(x_grid, rv_scipy.cdf(x_grid), label="True")
    # plt.plot(x_grid, rv_smoothed.cdf(x_grid), label="Smoothed")
    plt.title(f"{smooth=}")
    plt.legend()
    plt.show()

# %% Smoothing discrete
def smooth_kde(rv, smooth=None):
    to_class = "Cont" if isinstance(rv, Cont) else "Disc"
    x, a, b = rv.x, rv.a, rv.b
    rv_disc = rv.convert("Disc")
    kde = gaussian_kde(rv_disc.x, bw_method=smooth, weights=rv_disc.p)

    # y_smoothed = kde(x) + kde(a - (x - a)) + kde(b + (b - x))
    # return Cont(x=x, y=y_smoothed).convert(to_class=to_class)
    p_smoothed = kde(x) + kde(a - (x - a)) + kde(b + (b - x))
    return Disc(x=x, p=p_smoothed).convert(to_class=to_class)


x = np.sort(np.random.uniform(size=100))
p = np.random.uniform(size=len(x))
p /= p.sum()

rv_disc = Disc(x, p)
rv_disc_smooth = smooth_kde(rv_disc)

x_grid = np.linspace(x[0], x[-1], 1001)
# plt.plot(rv_disc.x, rv_disc.p, "o", label="Input")
# plt.plot(rv_disc_smooth.x, rv_disc_smooth.p, "o", label="Smooth")
plt.plot(x_grid, rv_disc.cdf(x_grid), label="Input")
plt.plot(x_grid, rv_disc_smooth.cdf(x_grid), label="Smooth")
plt.legend()
plt.show()

## Distance to uniform
from scipy.integrate import quad

rv_uniform = distrs.uniform(loc=x[0], scale=x[-1] - x[0])
quad(lambda t: np.abs(rv_uniform.cdf(t) - rv_disc.cdf(t)), x[0], x[-1], limit=100)
quad(
    lambda t: np.abs(rv_uniform.cdf(t) - rv_disc_smooth.cdf(t)), x[0], x[-1], limit=100
)
