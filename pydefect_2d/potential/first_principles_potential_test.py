# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile


@pytest.fixture
def fp_pot():
    return FirstPrinciplesPotentialProfile(
        z_length=10.0,
        defect_position_in_frac_coord=0.0,
        z_grid=[0., 2., 4., 6., 8.],
        xy_ave_potential=[-4., -2., 0., -2., -5.],
        num_grid_per_unit=3)


def test_macroscopic_average(fp_pot):
    assert fp_pot.macroscopic_average[0] == (-5.0-4.0-2.0) / 3
    fp_pot.num_grid_per_unit = 2
    assert fp_pot.macroscopic_average[0] == (-5.0-4.0) / 2