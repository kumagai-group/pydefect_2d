# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy.testing import assert_array_almost_equal

from pydefect_2d.potential.distribution import \
    make_gaussian_distribution, make_step_like_distribution, \
    rescale_distribution
from pydefect_2d.potential.grids import Grid

grid = Grid(10.0, 10)


def test_make_step_like_epsilon():
    actual = make_step_like_distribution(
        grid=grid, step_left=3.2, step_right=5.5, error_func_width=0.1)
    assert actual[3] == 0.0023388674905235884


def test_make_gaussian_epsilon():
    actual = make_gaussian_distribution(grid=grid,
                                        position=2.0,
                                        sigma=1.0)
    assert actual[1] == 0.6065306597126334


def test_rescale_distribution():
    actual = rescale_distribution(dist=np.array([1.0, 3.0]), average=1.0)
    expected = np.array([0.5, 1.5])
    assert_array_almost_equal(actual, expected)


