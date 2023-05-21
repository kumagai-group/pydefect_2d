# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pymatgen.io.vasp import Locpot

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile


def make_potential_profiler(locpot: Locpot,
                            defect_pos: float,
                            num_grid_per_atom: int):
    return FirstPrinciplesPotentialProfile(
        z_length=locpot.structure.lattice.c,
        defect_position_in_frac_coord=defect_pos,
        z_grid=locpot.get_axis_grid(ind=2),
        xy_ave_potential=locpot.get_average_along_axis(ind=2).tolist(),
        num_grid_per_unit=num_grid_per_atom)
