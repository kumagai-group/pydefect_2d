# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.analyzer.calc_results import CalcResults
from pydefect.cli.main_tools import parse_dirs
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_function import logger
from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.correction.isolated_gauss import CalcIsolatedGaussEnergy
from pydefect_2d.potential.one_d_potential import Fp1DPotential, OneDPotDiff, \
    PotDiffGradients, Gauss1DPotential
from pydefect_2d.potential.slab_model import GaussChargeModel, \
    CalcGaussChargePotential, SlabModel
from pydefect_2d.potential.slab_model_plotter import SlabModelPlotter
from pydefect_2d.util.utils import add_z_to_filename, show_x_values
from pydefect_2d.correction.gauss_energy import make_gauss_energies
from pydefect_2d.potential.grids import Grids, Grid

logger = get_logger(__name__)


def plot_volumetric_data(args):
    if "CHG" in args.filename:
        vol_data = Chgcar.from_file(args.filename)
    elif "LOCPOT" in args.filename:
        vol_data = Locpot.from_file(args.filename)
    else:
        raise ValueError

    ax = plt.gca()
    z_grid = vol_data.get_axis_grid(args.direction)
    values = vol_data.get_average_along_axis(ind=args.direction)

    if "CHG" in args.filename:
        values /= vol_data.structure.lattice.abc[args.direction]
        ax.set_ylabel("Charge density (e/Å)")
    if "LOCPOT" in args.filename:
        ax.set_ylabel("Potential (V)")

    ax.set_xlabel("Distance (Å)")
    ax.plot(z_grid, values, color="red")
    plt.ylim(args.y_range)
    if args.target_val:
        plt.hlines(args.target_val, z_grid[0], z_grid[-1], "blue",
                   linestyles='dashed')
    if args.target_val and args.z_guess:
        vals = show_x_values(np.array(z_grid), np.array(values), args.target_val,
                             args.z_guess)
        print(np.round(vals, 3))
    plt.savefig(f"{args.filename}.pdf")


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


def make_fp_1d_potential(args):
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=2)

    def _inner(_dir: Path):
        fp_potential = _make_fp_potential(_dir, perfect_pot)
        pot_grads = _make_pot_diff_grads(_dir, fp_potential, args.pot_dir)
        pot_grads.to_json_file()
        pot_grads.to_plot(plt.gca())
        plt.show()

        fp_potential.gauss_pos = pot_grads.gauss_pos_w_min_grad()
        logger.info(f"{_dir}: gauss pos is {fp_potential.gauss_pos}.")

        print(fp_potential.gauss_pos)
        fp_potential.to_json_file()

    parse_dirs(args.dirs, _inner, True, "fp1_d_potential.json")


def _make_fp_potential(_dir, perfect_pot) -> Fp1DPotential:
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
    return Fp1DPotential(grid, pot)


def _make_pot_diff_grads(_dir, fp_potential, pot_dir):
    grads, gaussian_pos = [], []
    defect_entry: DefectEntry = loadfn(_dir / "defect_entry.json")
    for gauss_1d_pot in _gauss_1d_pots(pot_dir):
        gauss_pot = deepcopy(gauss_1d_pot)
        gauss_pot.potential *= defect_entry.charge
        diff = OneDPotDiff(fp_pot=fp_potential, gauss_pot=gauss_pot)
        grads.append(diff.potential_diff_gradient)
        gaussian_pos.append(gauss_1d_pot.gauss_pos)
    return PotDiffGradients(grads, gaussian_pos)


def _gauss_1d_pots(pot_dir) -> List[Gauss1DPotential]:
    result = []
    for gauss_1d_pot in glob.glob(f'{pot_dir}/gauss1_d_potential*json'):
        result.append(loadfn(gauss_1d_pot))
    return sorted(result, key=lambda x: x.gauss_pos)


def make_gauss_model(args):
    """depends on the supercell size and defect position"""
    def _inner(_dir: Path):
        fp_potential: Fp1DPotential = loadfn(_dir / "fp1_d_potential.json")
        calc_results: CalcResults = loadfn(_dir / "calc_results.json")

        defect_z_pos = fp_potential.gauss_pos
        lat = calc_results.structure.lattice
        grids = Grids.from_z_grid(lat.matrix[:2, :2], args.diele_dist.dist.grid)

        gauss_charge = _make_gauss_charge_model(
            grids, args.std_dev, defect_z_pos, args.correction_dir)

        _make_gauss_potential(args.diele_dist, gauss_charge, args.multiprocess,
                              args.correction_dir)

        _make_isolated_gauss(args.diele_dist, gauss_charge,
                             args.k_max, args.k_mesh_dist,
                             args.correction_dir)

    parse_dirs(args.dirs, _inner, True, "gauss_charge_model.json")


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


def make_slab_model(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    def _inner(_dir: Path):
        fp_potential: Fp1DPotential = loadfn(_dir / "fp1_d_potential.json")
        defect_entry: DefectEntry = loadfn(_dir / "defect_entry.json")

        def _get_obj_from_corr_dir(filename: str):
            x, y = filename.split(".")
            filename = f"{x}_{fp_potential.gauss_pos:.3f}.{y}"
            try:
                return loadfn(args.correction_dir / filename)
            except FileNotFoundError:
                print(f"{filename} is not found.")
                raise

        gauss_charge_model = _get_obj_from_corr_dir("gauss_charge_model.json")
        gauss_charge_pot = _get_obj_from_corr_dir("gauss_charge_potential.json")
        isolated_energy = _get_obj_from_corr_dir("isolated_gauss_energy.json")

        logger.info(f"slab_model.json is being created.")
        slab_model = _make_slab_model(args.diele_dist,
                                      defect_entry,
                                      gauss_charge_model,
                                      gauss_charge_pot,
                                      fp_potential)
        _make_correction(isolated_energy, slab_model)
        # calc_results = loadfn(_dir / "calc_results.json")
        # _make_site_potential(args.perfect_calc_results, calc_results, slab_model)

    parse_dirs(args.dirs, _inner, True, "slab_model.json")


def _make_slab_model(diele_dist, defect_entry, gauss_charge_model,
                     gauss_charge_pot, fp_potential):
    result = SlabModel(diele_dist=diele_dist,
                       gauss_charge_model=gauss_charge_model,
                       gauss_charge_potential=gauss_charge_pot,
                       charge_state=defect_entry.charge,
                       fp_potential=fp_potential)
    result.to_json_file()
    SlabModelPlotter(plt, result)
    plt.savefig("potential_profile.pdf")
    return result


def _make_correction(isolated_gauss_energy, slab_model):
    squared_charge_state = slab_model.charge_state ** 2
    isolated_energy = isolated_gauss_energy.self_energy * squared_charge_state
    correction = Gauss2dCorrection(slab_model.charge_state,
                                   slab_model.electrostatic_energy,
                                   isolated_energy,
                                   slab_model.potential_diff)
    print(correction)
    correction.to_json_file()


# def _make_site_potential(perfect_calc_results, calc_results, slab_model):
#     sites = make_potential_sites(calc_results,
#                                  perfect_calc_results,
#                                  slab_model)
#     plt.clf()
#     plotter = SitePotentialMplPlotter(
#         title="atomic site potential", sites=sites)
#     plotter.construct_plot()

