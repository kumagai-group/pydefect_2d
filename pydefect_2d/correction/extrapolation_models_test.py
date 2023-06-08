# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import e

from pydefect_2d.correction.extrapolation_models import Extrapolation


@pytest.fixture
def extrapolation():
    return Extrapolation(muls=[1, 2, 3, 4],
                         electrostatic_energies=[2.331, 2.513, 2.596, 2.646])
#    electrostatic_energies=[e, e**2, e**3, e**4, e**5])

    #    asert extrapolation.extrapolation() ==
def test_extrapolation_error(extrapolation):
    extrapolation.to_plot(plt)
    plt.show()
