# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.make_epsilon_distribution import \
    make_epsilon_gaussian_dist, EpsilonGaussianDistribution
from pydefect_2d.potential.grids import Grid


@pytest.fixture
def epsilon():
    return EpsilonGaussianDistribution(grid=Grid(20, 2),
                                       electronic=[[1., 2.], [1., 2.], [1., 2.]],
                                       ionic=[[2., 3.], [2., 3.], [2., 3.]],
                                       center=5.001,
                                       sigma=0.2)


def test_epsilon_properties(epsilon):
    assert_almost_equal(epsilon.ion_clamped, [[2., 3.], [2., 3.], [2., 3.]])
    assert_almost_equal(epsilon.static, [[4., 6.], [4., 6.], [4., 6.]])
    assert_almost_equal(epsilon.effective, [[4., 6.], [4., 6.], [4., 6.]])


def test_epsilon_averages(epsilon):
    assert_almost_equal(epsilon.ave_ele, [1.5, 1.5, 1.5])
    assert_almost_equal(epsilon.ave_ion, [2.5, 2.5, 2.5])


def test_reciprocal_static(epsilon):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(epsilon.reciprocal_static[0], [10.-0.j, -2.-0.j])


def test_epsilon_str(epsilon):
    actual = epsilon.__str__()
    expected = """center: 5.00 Å
sigma: 0.20 Å
  pos (Å)    ε_inf_x    ε_inf_y    ε_inf_z    ε_ion_x    ε_ion_y    ε_ion_z    ε_0_x    ε_0_y    ε_0_z
     0.00       2.00       2.00       2.00       2.00       2.00       2.00     4.00     4.00     4.00
    10.00       3.00       3.00       3.00       3.00       3.00       3.00     6.00     6.00     6.00"""
    assert actual == expected


def test_epsilon_to_plot(epsilon):
    epsilon.to_plot(plt)
    plt.show()


def test_make_large_model():
    actual = make_epsilon_gaussian_dist(6.0, 3, [2/3.]*3, [4/3]*3,
                                        position=3., sigma=0.1, mul=2)
    expected = EpsilonGaussianDistribution(
        grid=Grid(6.0, 3, 2),
        electronic=[[0., 1., 1., 0., 0., 0.]] * 3,
        ionic=[[0., 2., 2., 0., 0., 0.]] * 3,
        center=3.,
        sigma=0.1)
    assert actual == expected
