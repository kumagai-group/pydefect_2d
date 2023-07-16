# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.distribution import GaussianDist
from pydefect_2d.potential.epsilon_distribution import \
    DielectricConstDist


@pytest.fixture
def gauss_epsilon():
    return DielectricConstDist(ave_ele=[3, 3, 1.5],
                               ave_ion=[1, 1, 0.5],
                               dist=GaussianDist(20, 2, center=10, sigma=0.2))


def test_epsilon_properties(gauss_epsilon):
    assert_almost_equal(gauss_epsilon.static[0], [1., 7.0])
    assert_almost_equal(gauss_epsilon.static[2], [1., 2.0])


def test_reciprocal_static(gauss_epsilon):
    """The returned complex array contains ``y(0), y(1),..., y(n-1)``, where
       ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``."""
    assert_almost_equal(gauss_epsilon.reciprocal_static[0], [8.-0.j, -6.-0.j])


def test_epsilon_to_plot(gauss_epsilon):
    gauss_epsilon.to_plot(plt.gca())
    plt.show()
