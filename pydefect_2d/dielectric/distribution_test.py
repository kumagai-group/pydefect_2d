# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy.testing import assert_array_almost_equal

from pydefect_2d.dielectric.distribution import \
    GaussianDist, ManualDist, StepDist


def test_gaussian_dist():
    actual = GaussianDist(length=10.0, num_grid=10, center=2.0,
                          in_plane_sigma=1.0, out_of_plane_sigma=1.0)
    assert actual.unscaled_in_plane_dist[1] == 0.6065306597126334


def test_step_dist():
    actual = StepDist(length=10.0, num_grid=10, center=4.35,
                      in_plane_width=2.3, in_plane_sigma=0.1,
                      out_of_plane_width=2.3, out_of_plane_sigma=0.1)
    assert round(actual.unscaled_in_plane_dist[3], 5) == 0.02275
    print(actual)


def test_diele_in_plane_scale():
    dist = ManualDist(length=1.0, num_grid=2,
                      unscaled_in_plane_dist_=np.array([1.0, 3.0]),
                      unscaled_out_of_plane_dist_=np.array([1.0, 3.0]))
    actual = dist.diele_in_plane_scale(ave_diele=2.0)
    expected = np.array([1.5, 2.5])
    assert_array_almost_equal(actual, expected)


def test_diele_out_of_plane_scale():
    dist = ManualDist(length=1.0, num_grid=3,
                      unscaled_in_plane_dist_=np.array([1.0, 1.0, 0.0]),
                      unscaled_out_of_plane_dist_=np.array([1.0, 1.0, 0.0]))
    actual = dist.diele_out_of_plane_scale(ave_diele=2.2)
    assert_array_almost_equal(actual, np.array([5.5, 5.5, 1.]), decimal=5)

