# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal

from pydefect_2d.potential.make_epsilon_distribution import EpsilonDistribution


@pytest.fixture
def epsilon():
    return EpsilonDistribution(grid=[0.0, 10.0],
                               ion_clamped=[[2., 3.], [2., 3.], [2., 3.]],
                               ionic=[[2., 3.], [2., 3.], [2., 3.]],
                               center=5.001)


def test_epsilon_properties(epsilon):
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


# def test_make_epsilon_distribution():
#     MakeEpsilonDistribution(lattice_constants=[10.0]*3,
#                             ave_ion_clamped_epsilon=[2.0, 3.0, 4.0],
#                             ave_ionic_epsilon=[2.0, 3.0, 4.0],
#                             model=
#                            )

