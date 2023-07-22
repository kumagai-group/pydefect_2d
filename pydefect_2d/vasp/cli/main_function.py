# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from math import ceil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.analyzer.defect_structure_info import DefectStructureInfo
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Chgcar, Locpot

from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.potential.distribution import GaussianDist, StepDist
from pydefect_2d.potential.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, FP1dPotential, SlabModel


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


def make_dielectric_distribution(args):
    """depends on the supercell size"""
    electronic = list(np.diag(args.unitcell.ele_dielectric_const))
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    grid = Grid(args.structure.lattice.c, args.num_grid)

    if args.type == "gauss":
        position = args.structure.lattice.c * args.position
        dist = GaussianDist.from_grid(grid, position, args.sigma)
    elif args.type == "step":
        dist = StepDist.from_grid(grid, args.step_left, args.step_right,
                                  args.error_func_width)
    else:
        raise ValueError

    diele = DielectricConstDist(electronic, ionic, dist)
    diele.to_json_file()


def _add_z_pos(filename: str, model: GaussChargeModel):
    x, y = filename.split(".")
    return f"{x}_{model.defect_z_pos_in_frac:.3}.{y}"


def make_gauss_charge_model(args):
    """depends on the supercell size and defect position"""
    dsi: DefectStructureInfo = args.defect_structure_info
    lat = dsi.shifted_final_structure.lattice
    grids = Grids.from_z_num_grid(lat.matrix[:2, :2], args.epsilon_dist.grid)

    model = GaussChargeModel(grids,
                             sigma=args.sigma,
                             defect_z_pos_in_frac=dsi.center[2])
    filename = _add_z_pos(model.json_filename, model)
    model.to_json_file(filename)


def calc_gauss_charge_potential(args):
    """depends on the supercell size and defect position"""
    potential = CalcGaussChargePotential(
        dielectric_const=args.epsilon_dist,
        gauss_charge_model=args.gauss_charge_model,
        multiprocess=args.multiprocess).potential
    filename = _add_z_pos(potential.json_filename, args.gauss_charge_model)
    potential.to_json_file(filename)


def isolated_gauss_energy(args):
    """depends on the supercell size, defect position"""
    isolated = IsolatedGaussEnergy(gauss_charge_model=args.gauss_charge_model,
                                   epsilon_z=args.epsilon_dist.static[2],
                                   k_max=args.k_max,
                                   k_mesh_dist=args.k_mesh_dist)
    print(isolated)
    filename = _add_z_pos(isolated.json_filename, args.gauss_charge_model)
    isolated.to_json_file(filename)


def make_fp_1d_potential(args):
    length = args.defect_locpot.structure.lattice.lengths[args.axis]
    grid_num = args.defect_locpot.dim[args.axis]

    defect_pot = args.defect_locpot.get_average_along_axis(ind=args.axis)
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=args.axis)

    try:
        # "-" is needed because the VASP potential is defined for electrons.
        pot = (-(defect_pot - perfect_pot)).tolist()
    except ValueError:
        print("The size of two LOCPOT files seems different.")
        raise

    FP1dPotential(Grid(length, grid_num), pot).to_json_file("fp_potential.json")


def _get_obj(dir_: Path, filename: str, defect_entry: DefectEntry):
    x, y = filename.split(".")
    filename = dir_ / f"{x}_{defect_entry.defect_center[2]:.3}.{y}"
    try:
        return loadfn(filename)
    except FileNotFoundError:
        print(f"{filename} is not found.")
        raise


def make_slab_model(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    d, de = args.correction_dir, args.defect_entry
    gauss_charge_model = _get_obj(d, "gauss_charge_model.json", de)
    gauss_charge_pot = _get_obj(d, "gauss_charge_potential.json", de)

    slab_model = SlabModel(charge=args.defect_entry.charge,
                           epsilon=args.epsilon_dist,
                           gauss_charge_model=gauss_charge_model,
                           gauss_charge_potential=gauss_charge_pot,
                           fp_potential=args.fp_potential)
    slab_model.to_json_file()
    ProfilePlotter(plt, slab_model)
    plt.savefig("potential_profile.pdf")


def make_correction(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    d = args.correction_dir
    isolated_gauss_energy = _get_obj(d, "isolated_gauss_energy.json",
                                     args.defect_entry)
    iso_e = isolated_gauss_energy.self_energy * args.slab_model.charge ** 2
    correction = Gauss2dCorrection(args.slab_model.charge,
                                   args.slab_model.electrostatic_energy,
                                   iso_e,
                                   args.slab_model.potential_diff)
    print(correction)
    correction.to_json_file()
