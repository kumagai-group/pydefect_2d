# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile


@pytest.fixture
def fp_pot():
    return FirstPrinciplesPotentialProfile(
        lattice_constants=[4., 4., 10.],
        num_grids=[8, 8, 20],
        defect_position_in_frac_coord=0.0,
        xy_ave_potential=[-4., -2., 0., -2., -5.])

