# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Chgcar, Locpot

from pydefect_2d.potential.make_epsilon_distribution import \
    make_epsilon_gaussian_dist
from pydefect_2d.potential.slab_model_info import CalcPotential, ProfilePlotter


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


def make_epsilon_distribution(args):
    grid = linspace(0., args.structure.lattice.c, args.num_grid, endpoint=False)
    clamped = list(np.diag(args.unitcell.ele_dielectric_const))
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    position = args.structure.lattice.c * args.position
    epsilon_distribution = make_epsilon_gaussian_dist(
        list(grid), clamped, ionic, position, args.sigma)
    epsilon_distribution.to_json_file()


def make_slab_gauss_model(args):
    de: DefectEntry = args.defect_entry
    lat = de.structure.lattice
    z_num_grid = len(args.epsilon_dist.static[0])
    x_num_grid = ceil(lat.a / lat.c * z_num_grid / 2) * 2
    y_num_grid = ceil(lat.b / lat.c * z_num_grid / 2) * 2

    defect_z_pos = lat.c * de.defect_center[2]

    fp_grid = args.defect_locpot.get_axis_grid(2)
    fp_defect_pot = args.defect_locpot.get_average_along_axis(ind=2)
    fp_perfect_pot = args.perfect_locpot.get_average_along_axis(ind=2)
    try:
        # minus is necessary because the VASP potential is for electrons.
        fp_pot = (-(fp_defect_pot - fp_perfect_pot)).tolist()
    except ValueError:
        print("The size of two LOCPOT files seems different.")
        raise

    model = CalcPotential(lattice_constants=[lat.a, lat.b, lat.c],
                          num_grids=[x_num_grid, y_num_grid, z_num_grid],
                          epsilon=args.epsilon_dist.static,
                          charge=de.charge,
                          sigma=args.sigma,
                          defect_z_pos=defect_z_pos,
                          fp_grid=fp_grid,
                          fp_xy_ave_potential=fp_pot)

    if args.calc_potential:
        model.real_charge
        model.real_potential
    model.to_json_file()
    model.to_plot(plt)
    plt.savefig("slab_gauss_model.pdf")
