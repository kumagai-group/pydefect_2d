# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy import linspace
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Outcar

from pydefect_2d.potential.make_epsilon_distribution import \
    make_gaussian_epsilon_distribution
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import SlabGaussModel
from pydefect_2d.vasp.potential.make_potential_profile import \
    make_potential_profiler


def make_epsilon_distribution(args):
    grid = linspace(0., args.structure.lattice.c, args.num_grid, endpoint=False)
    clamped = list(np.diag(args.unitcell.ele_dielectric_const))
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    epsilon_distribution = make_gaussian_epsilon_distribution(
        list(grid), clamped, ionic, args.position, args.sigma)
    epsilon_distribution.to_json_file()


def make_fp_pot(args):
    pot = make_potential_profiler(args.locpot, args.defect_entry.defect_center)
    pot.to_json_file()


def make_slab_gauss_model(args):
    de: DefectEntry = args.defect_entry
    lat = de.structure.lattice
    model = SlabGaussModel(lattice_constants=[lat.a, lat.b, lat.c],
                           epsilon=args.epsilon_dist.static,
                           charge=de.charge,
                           sigma=args.sigma,
                           defect_z_pos=de.defect_center[2])
    if args.calc_potential:
        model.real_potential
    model.to_json_file()
    plotter = ProfilePlotter(model)
    plotter.plt.savefig("slab_gauss_model.pdf")