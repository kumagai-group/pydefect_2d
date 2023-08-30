# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
import shutil
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.analyzer.defect_structure_info import DefectStructureInfo
from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.potential.calc_one_d_potential import Calc1DPotential, \
    OneDGaussChargeModel
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.dielectric.distribution import GaussianDist, StepDist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.correction.make_site_potential import make_potential_sites
from pydefect_2d.potential.one_d_potential import OneDPotDiff, \
    PotDiffGradients, Fp1DPotential
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel

logger = get_logger(__name__)


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
    z_grid = vol_data.get_axis_grid(args.direction)
    values = vol_data.get_average_along_axis(ind=args.direction)
#    if is_sum:
#        surface_area = np.prod(vol_data.structure.lattice.lengths[:2])
#        values *= surface_area
    ax.plot(z_grid, values, color="red")
    plt.savefig(f"{args.filename}.pdf")

