# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from numpy import linspace

from pydefect_2d.correction.epsilon_distribution import \
    make_epsilon_distribution


def test_make_epsilon_distribution():
    actual = make_epsilon_distribution(dielectric_constants=[2.0, 3.0, 4.0],
                                       grid=linspace(0, 2.0, 10),
                                       z_latt=2.0,
                                       slab_left=0.5, slab_right=1.5,
                                       error_func_width=0.1)

    expected = [1.0000000000007687, 1.000042761601475, 1.2160291905709464,
                1.9907889372729506, 1.9999999809801952, 1.9999999809801952,
                1.9907889372729506, 1.2160291905709473, 1.000042761601475,
                1.0000000000007687]
    assert actual[0] == expected
