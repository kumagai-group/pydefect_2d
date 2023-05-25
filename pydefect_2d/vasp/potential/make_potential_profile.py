# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pymatgen.io.vasp import Locpot

from pydefect_2d.potential.first_principles_potential import \
    FirstPrinciplesPotentialProfile


def make_potential_profiler(locpot: Locpot, defect_pos: float):
    return FirstPrinciplesPotentialProfile(
        lattice_constants=list(locpot.structure.lattice.lengths),
        num_grids=list(locpot.dim),
        defect_position_in_frac_coord=defect_pos,
        xy_ave_potential=locpot.get_average_along_axis(ind=2).tolist())
