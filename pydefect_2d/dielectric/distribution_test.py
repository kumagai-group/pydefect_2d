# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy.testing import assert_array_almost_equal

from pydefect_2d.dielectric.distribution import \
    GaussianDist, ManualDist, StepDist


def test_manual_dist():
    ManualDist(length=4.0, num_grid=4, manual_dist=np.array([1.0]*4))


def test_gaussian_dist():
    actual = GaussianDist(length=10.0, num_grid=10, center=2.0, sigma=1.0)
    assert actual.unscaled_dist[1] == 0.6065306597126334


def test_step_dist():
    actual = StepDist(length=10.0, num_grid=10, step_left=3.2, step_right=5.5,
                      error_func_width=0.1)
    assert actual.unscaled_dist[3] == 0.0023388674905235884


def test_diele_in_plane_scale():
    dist = ManualDist(length=1.0, num_grid=2, manual_dist=np.array([1.0, 3.0]))
    actual = dist.diele_in_plane_scale(ave_diele=1.0)
    expected = np.array([0.5, 1.5])
    assert_array_almost_equal(actual, expected)


def test_diele_out_of_plane_scale():
    dist = ManualDist(length=1.0, num_grid=3,
                      manual_dist=np.array([1.0, 1.0, 0.0]))
    actual = dist.diele_out_of_plane_scale(ave_diele=2.2)
    assert_array_almost_equal(actual, np.array([5.5, 5.5, 1.]), decimal=5)

