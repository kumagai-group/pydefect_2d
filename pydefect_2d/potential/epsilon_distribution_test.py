# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.epsilon_distribution import \
    EpsilonGaussianDistribution, EpsilonStepLikeDistribution, \
    scaling_z_direction
from pydefect_2d.potential.grids import Grid


@pytest.fixture
def gauss_epsilon():
    return EpsilonGaussianDistribution(grid=Grid(20, 2),
                                       ave_electronic_epsilon=[2, 2, 0.5],
                                       ave_ionic_epsilon=[1, 1, 0.2],
                                       center=10,
                                       sigma=0.2)


def test_epsilon_properties(gauss_epsilon):
    assert_almost_equal(gauss_epsilon.static[0], [1., 7.])
    assert_almost_equal(gauss_epsilon.static[2], [1., 5.66664416])


def test_reciprocal_static(gauss_epsilon):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(gauss_epsilon.reciprocal_static[0], [8.-0.j, -6.-0.j])


def test_epsilon_to_plot(gauss_epsilon):
    gauss_epsilon.to_plot(plt.gca())
    plt.show()


def test_scaling_z_direction():
    dist = [0.0, 1.0, 1.0, 0.0]
    actual = scaling_z_direction(np.array(dist), ave_diele=0.7)
    assert_almost_equal(actual, np.array([0., 4.6666442, 4.6666442, 0.]))