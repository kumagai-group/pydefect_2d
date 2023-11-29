# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import yaml
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.cli.main_tools import parse_dirs
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_plot_json import plot
from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.correction.gauss_energy import make_gauss_energies
from pydefect_2d.correction.isolated_gauss import CalcIsolatedGaussEnergy
from pydefect_2d.one_d.slab_model import OneDSlabModel
from pydefect_2d.three_d.slab_model import GaussChargeModel, \
    CalcGaussChargePotential
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotter
from pydefect_2d.util.utils import add_z_to_filename
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.dielectric.distribution import GaussianDist, StepDist, \
    PeriodicGaussianDist
from pydefect_2d.one_d.charge import OneDGaussChargeModel
from pydefect_2d.one_d.potential import Calc1DPotential, OneDFpPotential, \
    OneDPotDiff, PotDiffGradients, OneDGaussPotential
from pydefect_2d.three_d.grids import Grid, Grids

logger = get_logger(__name__)


def make_diele_dist(dist, args):
    ele = list(np.diag(args.unitcell.ele_dielectric_const))
    ion = list(np.diag(args.unitcell.ion_dielectric_const))

    slab_length = args.perfect_locpot.structure.lattice.c
    orig_num_grid = args.perfect_locpot.dim[2]
    num_grid = round(orig_num_grid / args.denominator)

    logger.info(f"Original #grid: {orig_num_grid}, #grid: {num_grid}")

    grid = Grid(slab_length, num_grid)
    if isinstance(args.center, list):
        if len(args.center) == 1:
            center = slab_length * args.center[0]
        else:
            center = [slab_length * c for c in args.center]
    else:
        center = slab_length * args.center

    diele = DielectricConstDist(ele, ion, dist(grid, center, args))
    diele.to_json_file()
    plot([diele.json_filename], plt.gca())
    plt.savefig("dielectric_const_dist.pdf")


def make_gauss_diele_dist(args):
    if len(args.center) == 1:
        def dist(grid, center, args_):
            return GaussianDist.from_grid(grid, center, args_.std_dev)
    else:
        def dist(grid, center, args_):
            return PeriodicGaussianDist.from_grid(grid, center, args_.std_dev)
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


def make_1d_gauss_models(args):
    left, right = sorted(args.range)
    n_grid = round((right - left) / args.mesh_distance) + 1
    gauss_pos = np.linspace(left, right, n_grid, endpoint=True)
    supercell = args.supercell_info.structure

    for pos in gauss_pos:
        charge_model = OneDGaussChargeModel(grid=args.diele_dist.dist.grid,
                                            std_dev=args.std_dev,
                                            surface=_xy_area(supercell),
                                            gauss_pos_in_frac=pos)
        filename = add_z_to_filename("1d_gauss_charge.json", pos)
        charge_model.to_json_file(filename)

        calc_1d_pot = Calc1DPotential(args.diele_dist, charge_model)
        filename = add_z_to_filename("1d_gauss_potential.json", pos)
        calc_1d_pot.potential.to_json_file(filename)


def _xy_area(structure):
    a, b = structure.lattice.matrix[:2, :2]
    return np.linalg.norm(np.cross(a, b))


def make_gauss_model_from_z(args):
    """depends on the supercell size and defect position"""
    lat = args.supercell_info.structure.lattice
    grids = Grids.from_z_grid(lat.matrix[:2, :2], args.diele_dist.dist.grid)
    for z_pos in args.z_pos:
        logger.info(f"At z={z_pos}...")
        filename = add_z_to_filename("gauss_charge_model.json", z_pos)
        if (args.correction_dir / filename).exists():
            logger.info(f"{filename} already exists, so skip.")
            continue

        gauss_charge = _make_gauss_charge_model(grids, args.std_dev, z_pos,
                                                args.correction_dir)

        _make_gauss_potential(args.diele_dist, gauss_charge, args.multiprocess,
                              args.correction_dir)

        _make_isolated_gauss(args.diele_dist, gauss_charge, args.k_max,
                             args.k_mesh_dist, args.correction_dir)


def make_gaussian_energies_from_args(args):
    gauss_energies = make_gauss_energies(args.correction_dir,
                                         args.z_range)
    gauss_energies.to_json_file()
    gauss_energies.to_plot(plt.gca())
    plt.savefig(fname="gauss_energies.pdf")
    plt.clf()


def make_1d_fp_potential(args):
    filename = "1d_fp_potential.json"
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=2)

    def _inner(_dir: Path):
        defect_entry: DefectEntry = loadfn(_dir / "defect_entry.json")
        if defect_entry.charge == 0:
            return

        fp_potential = _make_1d_fp_potential(_dir, perfect_pot)
        pot_grads = _make_pot_diff_grads(_dir, fp_potential, args.one_d_dir)
        pot_grads.to_json_file()
        pot_grads.to_plot(plt.gca())
        plt.savefig("pot_diff_gradients.pdf")

        fp_potential.gauss_pos = pot_grads.gauss_pos_w_min_grad()
        logger.info(f"{_dir}: gauss pos is {fp_potential.gauss_pos}.")

        print(fp_potential.gauss_pos)
        fp_potential.to_json_file(_dir / filename)

    parse_dirs(args.dirs, _inner, True, filename)


def _make_1d_fp_potential(_dir, perfect_pot) -> OneDFpPotential:
    locpot = Locpot.from_file(_dir / "LOCPOT")
    length = locpot.structure.lattice.lengths[2]
    grid_num = locpot.dim[2]
    grid = Grid(length, grid_num)
    defect_pot = locpot.get_average_along_axis(ind=2)
    try:
        # "-" is needed because the VASP potential is defined for electrons.
        pot = (-(defect_pot - perfect_pot)).tolist()
    except ValueError:
        print("The size of two LOCPOT files seems different.")
        raise
    return OneDFpPotential(grid, pot)


def _make_pot_diff_grads(_dir, fp_potential, pot_dir):
    grads, gaussian_pos = [], []
    defect_entry: DefectEntry = loadfn(_dir / "defect_entry.json")
    for one_d_gauss_pot in _1d_gauss_pots(pot_dir):
        gauss_pot = deepcopy(one_d_gauss_pot)
        gauss_pot.potential *= defect_entry.charge
        diff = OneDPotDiff(fp_pot=fp_potential, gauss_pot=gauss_pot)
        grads.append(diff.potential_diff_gradient)
        gaussian_pos.append(one_d_gauss_pot.gauss_pos)
    return PotDiffGradients(grads, gaussian_pos)


def _1d_gauss_pots(one_d_dir) -> List[OneDGaussPotential]:
    result = []
    for one_d_gauss_pot in glob.glob(f'{one_d_dir}/1d_gauss_potential*json'):
        result.append(loadfn(one_d_gauss_pot))
    return sorted(result, key=lambda x: x.gauss_pos)


def _make_gauss_charge_model(grids, std_dev, defect_z_pos, dir_):
    logger.info(f"GaussChargeModel is being created.")
    result = GaussChargeModel(grids, std_dev, defect_z_pos)
    filename = add_z_to_filename(result.json_filename, defect_z_pos)
    result.to_json_file(dir_ / filename)
    return result


def _make_gauss_potential(diele_dist, gauss_charge_model, multiprocess, dir_):
    logger.info(f"GaussChargePotential is being calculated.")
    result = CalcGaussChargePotential(
        dielectric_const=diele_dist,
        gauss_charge_model=gauss_charge_model,
        multiprocess=multiprocess).potential
    filename = add_z_to_filename(result.json_filename,
                                 gauss_charge_model.gauss_pos_in_frac)
    result.to_json_file(dir_ / filename)
    return result


def _make_isolated_gauss(diele_dist, gauss_charge_model, k_max, k_mesh_dist,
                         dir_):
    logger.info("Calculating isolated gauss charge self energy...")
    calculator = CalcIsolatedGaussEnergy(gauss_charge_model=gauss_charge_model,
                                         diele_const_dist=diele_dist,
                                         k_max=k_max,
                                         k_mesh_dist=k_mesh_dist)
    result = calculator.isolated_gauss_energy
    filename = add_z_to_filename(result.json_filename,
                                 gauss_charge_model.gauss_pos_in_frac)
    result.to_json_file(dir_ / filename)
    return result


def make_1d_slab_model(args):
    filename = "1d_slab_model.json"

    def _inner(dir_: Path):
        one_d_fp_potential = loadfn(dir_ / "1d_fp_potential.json")
        defect_entry: DefectEntry = loadfn(dir_ / "defect_entry.json")

        def _get_obj_from_corr_dir(filename: str):
            x, y = filename.split(".")
            filename = f"{x}_{one_d_fp_potential.gauss_pos:.3f}.{y}"
            try:
                return loadfn(args.one_d_dir / filename)
            except FileNotFoundError:
                print(f"{filename} is not found.")
                raise

        one_d_gauss_charge = _get_obj_from_corr_dir("1d_gauss_charge.json")
        one_d_gauss_pot = _get_obj_from_corr_dir("1d_gauss_potential.json")
        gauss_e = args.gauss_energies.get_gauss_energy(one_d_fp_potential.gauss_pos)

        one_d_gauss_charge.periodic_charges *= defect_entry.charge
        one_d_gauss_pot.potential *= defect_entry.charge
        gauss_e.isolated_energy *= defect_entry.charge ** 2
        gauss_e.periodic_energy *= defect_entry.charge ** 2

        slab_model = OneDSlabModel(charge_state=defect_entry.charge,
                                   diele_dist=args.diele_dist,
                                   one_d_gauss_charge=one_d_gauss_charge,
                                   one_d_gauss_potential=one_d_gauss_pot,
                                   one_d_fp_potential=one_d_fp_potential,
                                   gauss_energy=gauss_e)
        slab_model.to_json_file(dir_ / filename)
        correction = Gauss2dCorrection(slab_model.charge_state,
                                       slab_model.periodic_energy,
                                       slab_model.isolated_energy,
                                       slab_model.potential_diff)
        SlabModelPlotter(plt, slab_model)
        plt.savefig(dir_ / "potential_profile.pdf")
        correction.to_json_file(dir_ / "correction.json")

        if args.slab_center:
            shift = slab_model.get_xy_ave_potential(frac_coord=args.slab_center)
            d = {"shift_value": shift}
            Path(dir_ / "eigenvalue_shift.yaml").write_text(yaml.dump(d))

    parse_dirs(args.dirs, _inner, True, filename)
