# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from typing import Optional

from pydefect.analyzer.calc_results import CalcResults
from pydefect.cli.vasp.make_efnv_correction import make_sites
from vise.util.typing import Coords

from pydefect_2d.three_d.slab_model import SlabModel


def make_potential_sites(calc_results: CalcResults,
                         perfect_calc_results: CalcResults,
                         slab_model: SlabModel,
                         defect_coords: Optional[Coords] = None):
    charge = slab_model.charge_state
    sites, rel_coords, _ = \
        make_sites(calc_results, perfect_calc_results, defect_coords)

    for site, rel_coord in zip(sites, rel_coords):
        pot = slab_model.gauss_charge_potential.get_potential(rel_coord)
        site.pc_potential = pot * charge

    return sites

