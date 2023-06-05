# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Chgcar, Locpot

from pydefect_2d.potential.epsilon_distribution import \
    make_epsilon_gaussian_dist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.slab_model_info import CalcPotential, \
    GaussChargeModel, FP1dPotential, SlabModel
from pydefect_2d.potential.plotter import ProfilePlotter


def plot_volumetric_data(args):
    if "CHG" in args.filename:
        vol_data = Chgcar.from_file(args.filename)
        is_sum = True
    elif "LOCPOT" in args.filename:
        vol_data = Locpot.from_file(args.filename)
        is_sum = False
    else:
        raise ValueError

    ax = plt.gca()
    z_grid = vol_data.get_axis_grid(2)
    values = vol_data.get_average_along_axis(ind=2)
    if is_sum:
        surface_area = np.prod(vol_data.structure.lattice.lengths[:2])
        values *= surface_area
    ax.plot(z_grid, values, color="red")
    plt.savefig(f"{args.filename}.pdf")


def _add_mul(filename: str, mul):
    if mul == 1:
        return filename
    x, y = filename.split(".")
    return f"{x}_{mul}.{y}"


def make_epsilon_distributions(args):
    clamped = np.diag(args.unitcell.ele_dielectric_const)
    electronic = list(clamped - 1.)
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    position = args.structure.lattice.c * args.position

    for mul in args.muls:
        epsilon_distribution = make_epsilon_gaussian_dist(
            args.structure.lattice.c, args.num_grid, electronic, ionic,
            position, args.sigma, mul)
        filename = _add_mul(epsilon_distribution._json_filename, mul)
        epsilon_distribution.to_json_file(filename)


def make_gauss_charge_models(args):
    de: DefectEntry = args.defect_entry
    lat = de.structure.lattice
    z_num_grid = args.epsilon_dist.grid.num_grid
    x_num_grid = ceil(lat.a / lat.c * z_num_grid / 2) * 2
    y_num_grid = ceil(lat.b / lat.c * z_num_grid / 2) * 2

    defect_z_pos = lat.c * de.defect_center[2]

    for mul in args.muls:
        grids = Grids([Grid(lat.a, x_num_grid, mul),
                       Grid(lat.b, y_num_grid, mul),
                       Grid(lat.c, z_num_grid, mul)])

        model = GaussChargeModel(grids,
                                 charge=de.charge,
                                 sigma=args.sigma,
                                 defect_z_pos=defect_z_pos)
        filename = _add_mul("gauss_charge_model.json", mul)
        model.to_json_file(filename)


def calc_potential(args):
    calc_pot = CalcPotential(epsilon=args.epsilon_dist,
                             gauss_model=args.gauss_model,
                             multiprocess=args.multiprocess)
    filename = _add_mul("potential.json", args.epsilon_dist.grid.mul)
    calc_pot.potential.to_json_file(filename)


def make_fp_1d_potential(args):
    length = args.defect_locpot.structure.lattice.lengths[args.axis]
    grid_num = args.defect_locpot.dim[args.axis]

    defect_pot = args.defect_locpot.get_average_along_axis(ind=args.axis)
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=args.axis)

    try:
        # minus is necessary because the VASP potential is for electrons.
        pot = (-(defect_pot - perfect_pot)).tolist()
    except ValueError:
        print("The size of two LOCPOT files seems different.")
        raise

    FP1dPotential(Grid(length, grid_num), pot).to_json_file("fp_potential.json")


def potential_prof(args):
    slab_model = SlabModel(epsilon=args.epsilon,
                           charge=args.gauss_model,
                           potential=args.potential,
                           fp_potential=args.fp_potential)
    ele_energy = slab_model.to_electrostatic_energy
    filename = _add_mul("electrostatic_energy.json", args.epsilon.grid.mul)
    ele_energy.to_json_file(filename)

    ProfilePlotter(plt, slab_model)
    filename = _add_mul("potential.pdf", args.epsilon.grid.mul)
    plt.savefig(filename)
