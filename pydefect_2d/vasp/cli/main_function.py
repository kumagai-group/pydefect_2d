# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from numpy import linspace

from pydefect_2d.potential.make_epsilon_distribution import \
    make_gaussian_epsilon_distribution


def make_epsilon_distribution(args):
    grid = linspace(0., args.structure.lattice.c, args.num_grid, endpoint=False)
    clamped = list(np.diag(args.unitcell.ele_dielectric_const))
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    epsilon_distribution = make_gaussian_epsilon_distribution(
        list(grid), clamped, ionic, args.position, args.sigma)
    print(epsilon_distribution)
    epsilon_distribution.to_json()
