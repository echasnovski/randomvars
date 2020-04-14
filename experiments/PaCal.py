from time import time

from pacal import *
import numpy as np

# This only creates a "plan" on how to compute output distribution. Calling
# some method of output seems to actually do computation.
distr = UniformDistr(0, 1) + UniformDistr(1, 4)
start = time()
distr.pdf(0.2)
end = time()
print(end - start)
distr.plot()

N = NormalDistr()
L = N + N
start = time()
L.pdf(0.2)
end = time()
print(end - start)

L = NormalDistr() + UniformDistr(-10, 10)
start = time()
L.pdf(0.2)
end = time()
print(end - start)
L.plot()

N = NormalDistr()
from scipy.stats.distributions import norm
np.random.seed(101)
x = np.random.random(10000)
%timeit N.pdf(x)
%timeit norm.pdf(x)
np.abs(N.pdf(x) - norm.pdf(x))

# Custom function in density currently don't really work
def my_density(x):
    return np.sqrt(1 - np.clip(x, -1, 1) ** 2)


my_distr = FunDistr(my_density)
# my_distr = FunDistr(my_density, breakPoints=np.array([-1, 1]))
my_distr.pdf(0.5)
my_distr_sum = my_distr + my_distr
my_distr_sum.plot()

TrapezoidalDistr(a=1, b=2, c=1, d=1).plot()
