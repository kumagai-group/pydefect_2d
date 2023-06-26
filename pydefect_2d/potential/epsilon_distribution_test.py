# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.epsilon_distribution import \
    EpsilonGaussianDistribution, EpsilonStepLikeDistribution
from pydefect_2d.potential.grids import Grid


@pytest.fixture
def gauss_epsilon():
    return EpsilonGaussianDistribution(grid=Grid(20, 2),
                                       ave_electronic_epsilon=[2, 2, 2],
                                       ave_ionic_epsilon=[1, 1, 1],
                                       center=10,
                                       sigma=0.2)


def test_epsilon_properties(gauss_epsilon):
    assert_almost_equal(gauss_epsilon.electronic[0], [0., 4.])
    assert_almost_equal(gauss_epsilon.ionic[0], [0., 2.])

    assert_almost_equal(gauss_epsilon.ion_clamped[0], [1., 5.])
    assert_almost_equal(gauss_epsilon.static[0], [1., 7.])
    assert_almost_equal(gauss_epsilon.effective[0], [float("inf"), 5.+5.**2/2.])


def test_reciprocal_static(gauss_epsilon):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(gauss_epsilon.reciprocal_static[0], [8.-0.j, -6.-0.j])


def test_epsilon_to_plot(gauss_epsilon):
    gauss_epsilon.to_plot(plt)
    plt.show()


