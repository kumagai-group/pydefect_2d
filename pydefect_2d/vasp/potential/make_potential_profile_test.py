# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pymatgen.io.vasp import Locpot

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile
from pydefect_2d.vasp.potential.make_potential_profile import \
    make_potential_profiler


def test_make_potential_profiler(test_files):
    locpot = Locpot.from_file(test_files / "model_LOCPOT")
    actual = make_potential_profiler(locpot=locpot,
                                     defect_pos=0.0,
                                     num_grid_per_unit=2)
    expected = FirstPrinciplesPotentialProfile(
        z_length=2.0,
        defect_position_in_frac_coord=0.0,
        z_grid=[0.0, 0.5, 1.0, 1.5],
        xy_ave_potential=[7.5, 23.5, 39.5, 55.5],
        num_grid_per_unit=2)
    assert actual == expected
