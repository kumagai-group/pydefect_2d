# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.make_epsilon_distribution import EpsilonDistribution, \
    make_gaussian_epsilon_distribution, make_large_model


@pytest.fixture
def epsilon():
    return EpsilonDistribution(grid=[0.0, 10.0],
                               electronic=[[1., 2.], [1., 2.], [1., 2.]],
                               ionic=[[2., 3.], [2., 3.], [2., 3.]],
                               center=5.001)


def test_epsilon_properties(epsilon):
    assert_almost_equal(epsilon.ion_clamped, [[2., 3.], [2., 3.], [2., 3.]])
    assert_almost_equal(epsilon.static, [[4., 6.], [4., 6.], [4., 6.]])
    assert_almost_equal(epsilon.effective, [[4., 6.], [4., 6.], [4., 6.]])


def test_epsilon_str(epsilon):
    actual = epsilon.__str__()
    expected = """center: 5.00 Å
  pos (Å)    ε_inf_x    ε_inf_y    ε_inf_z    ε_ion_x    ε_ion_y    ε_ion_z    ε_0_x    ε_0_y    ε_0_z
     0.00       2.00       2.00       2.00       2.00       2.00       2.00     4.00     4.00     4.00
    10.00       3.00       3.00       3.00       3.00       3.00       3.00     6.00     6.00     6.00"""
    assert actual == expected


def test_epsilon_to_plot(epsilon):
    epsilon.to_plot(plt)
    plt.show()


def test_make_epsilon_distribution(mocker):
    mock = mocker.patch("pydefect_2d.potential.make_epsilon_distribution.make_gaussian_distribution")
    mock.return_value = [0.0, 1.0]
    actual = make_gaussian_epsilon_distribution(
        grid=[0.0, 1.0],
        ave_ion_clamped_epsilon=[2.0, 4.0, 6.0],
        ave_ionic_epsilon=[2.0, 3.0, 4.0],
        position=2.0,
        sigma=1.0)
    expected = EpsilonDistribution(
        grid=[0.0, 1.0],
        ion_clamped=[[1.0, 3.0], [1.0, 7.0], [1.0, 11.0]],
        ionic=[[0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
        center=2.0)
    assert actual == expected


def test_make_large_model():
    e_dist = EpsilonDistribution(grid=[0.0, 2.0, 4.0],
                                 ion_clamped=[[1., 2., 2.]]*3,
                                 ionic=[[0., 2., 2.]]*3,
                                 center=3.)
    actual = make_large_model(epsilon_dist=e_dist, mul=2)
    expected = EpsilonDistribution(grid=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                                   ion_clamped=[[1., 2., 2., 1., 1., 1.]] * 3,
                                   ionic=[[0., 2., 2., 0., 0., 0.]] * 3,
                                   center=3.)
    assert actual == expected
