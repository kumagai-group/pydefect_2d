# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from vise.util.logger import get_logger

from pydefect_2d.cli.main_plot_json import plot
from pydefect_2d.util.utils import add_z_to_filename
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.dielectric.distribution import GaussianDist, StepDist
from pydefect_2d.one_d.one_d_charge import OneDGaussChargeModel
from pydefect_2d.one_d.one_d_potential import Calc1DPotential
from pydefect_2d.potential.grids import Grid

logger = get_logger(__name__)


def make_diele_dist(dist, args):
    ele = list(np.diag(args.unitcell.ele_dielectric_const))
    ion = list(np.diag(args.unitcell.ion_dielectric_const))

    slab_length = args.perfect_locpot.structure.lattice.c
    orig_num_grid = args.perfect_locpot.dim[2]
    num_grid = round(orig_num_grid / args.denominator)

    logger.info(f"Original #grid: {orig_num_grid}, #grid: {num_grid}")
#    logger.info(f"The number of grid is set to {num_grid}")

    center = slab_length * args.center
    grid = Grid(slab_length, num_grid)

    diele = DielectricConstDist(ele, ion, dist(grid, center, args))
    diele.to_json_file()
    plot([diele.json_filename], plt.gca())
    plt.savefig("dielectric_const_dist.pdf")


def make_gauss_diele_dist(args):
    def dist(grid, center, args_):
        return GaussianDist.from_grid(grid, center, args_.std_dev)

    make_diele_dist(dist, args)


def make_step_diele_dist(args):
    def dist(grid, center, args_):
        width_z = args_.step_width_z or args_.step_width

        return StepDist.from_grid(
            grid=grid, center=center,
            in_plane_width=args_.step_width,
            out_of_plane_width=width_z,
            sigma=args_.std_dev)

    return make_diele_dist(dist, args)


make_gauss_charge_model_msg = \
    """defect_structure_info.json or a set of (supercell_info.json, defect_pos) 
need to be specified."""


def make_1d_gauss_models(args):
    left, right = sorted(args.range)
    n_grid = round((right - left) / args.mesh_distance) + 1
    gauss_pos = np.linspace(left, right, n_grid, endpoint=True)
    supercell = args.supercell_info.structure

    for pos in gauss_pos:
        filename = add_z_to_filename("gauss1_d_potential.json", pos)
        if Path(filename).exists():
            logger.info(f"Because {filename} exists, so skip.")
            continue

        charge_model = OneDGaussChargeModel(grid=args.diele_dist.dist.grid,
                                            std_dev=args.std_dev,
                                            surface=_xy_area(supercell),
                                            gauss_pos_in_frac=pos)
        calc_1d_pot = Calc1DPotential(args.diele_dist, charge_model)
        calc_1d_pot.potential.to_json_file(filename)


def _xy_area(structure):
    a, b = structure.lattice.matrix[:2, :2]
    return np.linalg.norm(np.cross(a, b))

