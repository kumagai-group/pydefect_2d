# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy.testing import assert_array_almost_equal

from pydefect_2d.util.utils import get_z_from_filename, show_x_values


def test_get_z_from_filename():
    assert get_z_from_filename("isolated_gauss_energy_0.370.json") == 0.37


def test_show_x_values():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ys = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
    actual = show_x_values(xs, ys, given_y=0.5, x_guess=[1, 3])
    assert_array_almost_equal(actual, [0.5, 3.5])
