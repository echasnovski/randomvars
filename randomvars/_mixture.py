""" Code for mixture random variable
"""

import warnings

import numpy as np

from randomvars._continuous import Cont
from randomvars._discrete import Disc


class Mixt:
    """Mixture random variable

    Class for mixture random variable which has a mixture distribution of
    continuous (of class `Cont`) and discrete (of class `Dict`) random
    variables with some predefined weights. Basically, this might be treated as
    a random variable which generates one sample value in two steps:

    - First, select a part from which sample will be drawn. Continuous part
      might be selected with `weight_cont` probability, discrete - with
      `weight_disc` (equal to `1 - weight_cont`) probability.
    - Second, sample a value from selected distribution.

    There are three ways to create instance of `Mixt` class:

    1. Directly supply continuous (`cont`) and discrete (`disc`) random
    variables along with weight of **continuous** part (`weight_cont`):
    ```
        from randomvars import Cont, Disc
        my_cont = Cont(x=[0, 1], y=[1, 1])
        my_disc = Disc(x=[0, 1], prob=[0.25, 0.75])

        # This is an equal weighted mixture of continuous uniform and bernoulli
        # random variables
        my_mixt = Mixt(cont=my_cont, disc=my_disc, weight_cont=0.5)
        print(my_mixt)

        # `Mixt` can hold only one part when it has full weight and the other
        # part is `None`.
        my_mixt_2 = Mixt(cont=my_cont, disc=None, weight_cont=1.0)
        print(my_mixt_2)
    ```
    2. Use `Mixt.from_rv()`:
    ```

    ```
    3. Use `Mixt.from_sample()`:
    ```

    ```
    """

    def __init__(self, cont, disc, weight_cont):
        self._cont, self._disc, self._weight_cont = self._impute_init_args(
            cont, disc, weight_cont
        )
        self._weight_disc = 1.0 - self._weight_cont

    @staticmethod
    def _impute_init_args(cont, disc, weight_cont):
        # Impute `weight_cont`
        try:
            weight_cont = float(weight_cont)
        except ValueError:
            raise ValueError("`weight_cont` should be a number.")

        if (weight_cont < 0) or (weight_cont > 1):
            raise ValueError("`weight_cont` should be between 0 and 1 (inclusively).")

        # Impute `cont`
        if cont is None:
            if weight_cont > 0:
                raise ValueError("`cont` can't be `None` if `weight_cont` is above 0.")
        elif not isinstance(cont, Cont):
            raise ValueError("`cont` should be object of `Cont` or `None`.")

        # Impute `disc`
        if disc is None:
            if weight_cont < 1:
                raise ValueError("`disc` can't be `None` if `weight_cont` is below 1.")
        elif not isinstance(disc, Disc):
            raise ValueError("`disc` should be object of `Disc` or `None`.")

        return cont, disc, weight_cont

    def __str__(self):
        return (
            "Mixture RV:\n"
            f"Cont (weight = {self.weight_cont}): {self.cont}\n"
            f"Disc (weight = {self.weight_disc}): {self.disc}"
        )

    @property
    def cont(self):
        return self._cont

    @property
    def disc(self):
        return self._disc

    @property
    def weight_cont(self):
        return self._weight_cont

    @property
    def weight_disc(self):
        return self._weight_disc
