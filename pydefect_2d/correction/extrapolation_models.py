# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from scipy.constants import e
from scipy.optimize import curve_fit


#def komsa_fit(x, c0, c1, c2, c3, d):
def komsa_fit(x, c0, c1, c2, d):
    return c0 + c1*x + c2*x**2
#    return c0 + c1*x + c2*x**2 + d*e**(-c3*x)


@dataclass
class Extrapolation:
    muls: List[int]
    electrostatic_energies: List[float]

    @property
    def inv_mul(self) -> np.array:
        return 1 / np.array(self.muls)

    @property
    def inv_mul_range(self) -> np.array:
        return np.linspace(0, max(self.inv_mul), 30)

    @cached_property
    def f_and_coeffs(self):
        try:
            x = self.inv_mul
            y = self.electrostatic_energies

            popt, pocv = curve_fit(komsa_fit, x, y)

            def f(x):
                return komsa_fit(x, *popt)

            return f, popt

        except TypeError as e:
            print(e)

    @property
    def c0(self):
        return self.f_and_coeffs[1][0]

    @property
    def f(self):
        return self.f_and_coeffs[0]

    def to_plot(self, plt):
#        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        plt.scatter(self.inv_mul, self.electrostatic_energies)
        ax = plt.gca()
        print(self.inv_mul_range)
        print(self.f(self.inv_mul_range))
        ax.plot(self.inv_mul_range, self.f(self.inv_mul_range))
        ax.legend()


