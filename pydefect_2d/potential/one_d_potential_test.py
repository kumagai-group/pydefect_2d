# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest

from pydefect_2d.potential.grids import Grid
from pydefect_2d.potential.one_d_potential import OneDPotDiff, \
    Gauss1DPotential, Fp1DPotential

grid = Grid(10., 4)


@pytest.fixture
def fp_pot():
    return Fp1DPotential(grid, potential=np.array([0.2, 0.4, 0.6, 0.8]))


@pytest.fixture
def gauss_pot():
    # gauss pos locate at grid index 3.
    return Gauss1DPotential(
        grid, np.array([0.0, -0.1, 0.0, 0.1]), gauss_pos=0.5)


def test_one_d_pot_diff(fp_pot, gauss_pot):
    one_d_pot_diff = OneDPotDiff(fp_pot, gauss_pot)
    actual = round(one_d_pot_diff.potential_diff_gradient, 2)
    expected = ((0.1 * 2 - 0.4) - (-0.1 * 2 - 0.2)) / (2 * 2.5)
    assert actual == expected




