import numpy as np
import scipy.stats as ss

from randomvars import Disc


class Binom(Disc):
    def __init__(self, n, prob):
        self._n = n
        self._prob = prob

        x = np.arange(n + 1)
        p = ss.binom(n=n, p=prob).pmf(x)

        super().__init__(x=x, p=p)

    @property
    def params(self):
        return {"n": self._n, "prob": self._prob}

    @property
    def n(self):
        return self._n

    @property
    def prob(self):
        return self._prob


n = 10
p = 0.5
b_ref = Disc.from_rv(ss.binom(n=n, p=p))
b = Binom(n, p)

# `False` because they are objects of different classes
b == b_ref

# Correct `params` is achieved because of approach with custom property
b.params

# Parent attributes are accessible
(b.x, b.p)

# Correct checks for equality
b2 = Binom(n, p)
b == b2

# Correct statistical properties
np.allclose(b.cdf(b.x), ss.binom(n=n, p=p).cdf(b.x))
