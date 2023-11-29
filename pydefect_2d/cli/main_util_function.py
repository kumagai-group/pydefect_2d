# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.analyzer.calc_results import CalcResults
from pydefect.cli.main_tools import parse_dirs
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_function import _make_gauss_charge_model, \
    _make_gauss_potential, _make_isolated_gauss
from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.dielectric.make_extended_diele_dist import AddVacuum, \
    RepeatDieleDist
from pydefect_2d.one_d.potential import OneDFpPotential
from pydefect_2d.three_d.slab_model import SlabModel
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotter
from pydefect_2d.util.utils import show_x_values
from pydefect_2d.three_d.grids import Grids


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


def make_gauss_model(args):
    """depends on the supercell size and defect position"""
    def _inner(_dir: Path):
        fp_potential: OneDFpPotential = loadfn(_dir / "fp_1d_potential.json")
        calc_results: CalcResults = loadfn(_dir / "calc_results.json")

        defect_z_pos = fp_potential.gauss_pos
        lat = calc_results.structure.lattice
        grids = Grids.from_z_grid(lat.matrix[:2, :2], args.orig_diele_dist.dist.grid)

        gauss_charge = _make_gauss_charge_model(
            grids, args.std_dev, defect_z_pos, args.correction_dir)

        _make_gauss_potential(args.orig_diele_dist, gauss_charge, args.multiprocess,
                              args.correction_dir)

        _make_isolated_gauss(args.orig_diele_dist, gauss_charge,
                             args.k_max, args.k_mesh_dist,
                             args.correction_dir)

    parse_dirs(args.dirs, _inner, True, "gauss_charge_model.json")


def make_slab_model(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    def _inner(_dir: Path):
        fp_potential: OneDFpPotential = loadfn(_dir / "fp_1d_potential.json")
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
        slab_model = _make_slab_model(args.orig_diele_dist,
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
                       charge_state=defect_entry.charge_state,
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


def add_vacuum(args):
    added_length = args.length - args.diele_dist.dist.length
    new_diele_dist = AddVacuum(args.diele_dist, added_length).diele_const_dist
    new_diele_dist.to_json_file(f"dielectric_const_dist_{args.length}Å.json")


def repeat_diele_dist(args):
    new_diele_dist = RepeatDieleDist(args.diele_dist, args.mul).diele_const_dist
    new_diele_dist.to_json_file(f"dielectric_const_dist_x{args.mul}.json")


def make_eigenvalue_shift(args):

    new_diele_dist.to_json_file(f"dielectric_const_dist_x{args.mul}.json")
