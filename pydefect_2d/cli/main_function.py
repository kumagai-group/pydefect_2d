# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.cli.main_tools import parse_dirs
from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.io.vasp import Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_plot_json import plot
from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.correction.make_site_potential import make_potential_sites
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.dielectric.distribution import GaussianDist, StepDist
from pydefect_2d.potential.calc_one_d_potential import Calc1DPotential, \
    OneDGaussChargeModel
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.one_d_potential import OneDPotDiff, \
    PotDiffGradients, Fp1DPotential, Gauss1DPotential
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel

logger = get_logger(__name__)


def make_diele_dist(dist, args):
    ele = list(np.diag(args.unitcell.ele_dielectric_const))
    ion = list(np.diag(args.unitcell.ion_dielectric_const))
    slab_length = args.perfect_slab.lattice.c
    center = slab_length * args.center
    grid = Grid.from_mesh_distance(slab_length, args.mesh_distance)

    diele = DielectricConstDist(ele, ion, dist(grid, center, args))
    diele.to_json_file()
    plot(diele.json_filename, diele)


def make_gauss_diele_dist(args):
    def dist(grid, center, args_):
        return GaussianDist.from_grid(grid, center, args_.sigma)
    make_diele_dist(dist, args)


def make_step_diele_dist(args):
    def dist(grid, center, args_):
        return StepDist.from_grid(
            grid, center, args_.step_width, args.error_func_width)
    make_diele_dist(dist, args)


def _add_z_pos(filename: str, pos_in_frac: float):
    x, y = filename.split(".")
    return f"{x}_{pos_in_frac:.3f}.{y}"


make_gauss_charge_model_msg = \
    """defect_structure_info.json or a set of (supercell_info.json, defect_pos) 
need to be specified."""


def make_1d_gauss_models(args):
    left, right = sorted(args.range)
    n_grid = round((right - left) / args.mesh_distance) + 1
    gauss_pos = np.linspace(left, right, n_grid, endpoint=True)
    supercell = args.supercell_info.structure

    for pos in gauss_pos:
        filename = _add_z_pos("gauss1_d_potential.json", pos)
        if Path(filename).exists():
            logger.info(f"Because {filename} exists, so skip.")
            continue

        charge_model = OneDGaussChargeModel(grid=args.diele_dist.dist.grid,
                                            sigma=args.sigma,
                                            surface=_xy_area(supercell),
                                            gauss_pos_in_frac=pos)
        calc_1d_pot = Calc1DPotential(args.diele_dist, charge_model)
        calc_1d_pot.potential.to_json_file(filename)


def _xy_area(structure):
    a, b = structure.lattice.matrix[:2, :2]
    return np.linalg.norm(np.cross(a, b))


def make_fp_1d_potential(args):
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=2)

    def _inner(_dir: Path):
        fp_potential = _make_fp_potential(_dir, perfect_pot)
        pot_grads = _make_pot_diff_grads(_dir, fp_potential, args.pot_dir)
        pot_grads.to_json_file()
        pot_grads.to_plot(plt.gca())
        plt.show()

        fp_potential.gauss_positions = pot_grads.gauss_pos_w_min_grad()
        logger.info(f"{_dir}: gauss pos is {fp_potential.gauss_positions}.")
        fp_potential.to_json_file()

    parse_dirs(args.dirs, _inner, True, "fp1_d_potential.json")


def _make_fp_potential(_dir, perfect_pot):
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

    lat = args.supercell_info.structure.lattice
    grids = Grids.from_z_grid(lat.matrix[:2, :2], args.diele_dist.dist.grid)

    gauss_charge = _make_gauss_charge_model(
        grids, args.sigma, args.defect_z_pos)
    potential = _make_gauss_potential(
        args.diele_dist, gauss_charge, args.multiprocess)
    logger.info("Calculating isolated gauss charge self energy...")
    isolated = _make_isolated_gauss(
        args.diele_dist, gauss_charge, args.k_max, args.k_mesh_dist)
    return gauss_charge, potential, isolated


def _make_gauss_charge_model(grids, sigma, defect_z_pos):
    logger.info(f"GaussChargeModel is being created at {defect_z_pos}")
    result = GaussChargeModel(grids, sigma, defect_z_pos)
    filename = _add_z_pos(result.json_filename, defect_z_pos)
    result.to_json_file(filename)
    return result


def _make_gauss_potential(diele_dist, gauss_charge_model, multiprocess):
    pos = gauss_charge_model.gauss_pos_in_frac
    logger.info(f"GaussChargePotential is being calculated at {pos}")
    result = CalcGaussChargePotential(
        dielectric_const=diele_dist,
        gauss_charge_model=gauss_charge_model,
        multiprocess=multiprocess).potential
    filename = _add_z_pos(result.json_filename,
                          gauss_charge_model.gauss_pos_in_frac)
    result.to_json_file(filename)
    return result


def _make_isolated_gauss(diele_dist, gauss_charge_model, k_max, k_mesh_dist):
    result = IsolatedGaussEnergy(gauss_charge_model=gauss_charge_model,
                                 diele_const_dist=diele_dist,
                                 k_max=k_max,
                                 k_mesh_dist=k_mesh_dist)
    filename = _add_z_pos(result.json_filename,
                          gauss_charge_model.gauss_pos_in_frac)
    result.to_json_file(filename)
    return result


def make_slab_model(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """

    gauss_pos = args.fp_potential.gauss_pos

    def _get_obj_from_corr_dir(filename: str):
        x, y = filename.split(".")
        filename = f"{x}_{gauss_pos:.3f}.{y}"
        try:
            return loadfn(args.correction_dir / filename)
        except FileNotFoundError:
            print(f"{filename} is not found.")
            raise

    gauss_charge_model = _get_obj_from_corr_dir("gauss_charge_model.json")
    gauss_charge_pot = _get_obj_from_corr_dir("gauss_charge_potential.json")
    isolated_gauss_energy = _get_obj_from_corr_dir("isolated_gauss_energy.json")

    calc_results = loadfn(args.dir / "calc_results.json")
    defect_entry: DefectEntry = loadfn(args.dir / "defect_entry.json")

    slab_model = _make_slab_model(args.diele_dist,
                                  defect_entry,
                                  gauss_charge_model,
                                  gauss_charge_pot,
                                  args.fp_potential)
    _make_correction(isolated_gauss_energy, slab_model)
    _make_site_potential(args.perfect_calc_results, calc_results, slab_model)


def _make_slab_model(diele_dist, defect_entry, gauss_charge_model,
                     gauss_charge_pot, fp_potential):
    result = SlabModel(diele_dist=diele_dist,
                       gauss_charge_model=gauss_charge_model,
                       gauss_charge_potential=gauss_charge_pot,
                       charge_state=defect_entry.charge,
                       fp_potential=fp_potential)
    result.to_json_file()
    ProfilePlotter(plt, result)
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


def _make_site_potential(perfect_calc_results, calc_results, slab_model):
    sites = make_potential_sites(calc_results,
                                 perfect_calc_results,
                                 slab_model)
    plotter = SitePotentialMplPlotter(
        title="atomic site potential", sites=sites)
    plotter.construct_plot()
    plotter.plt.savefig(fname="atomic_site_potential.pdf")
    plotter.plt.clf()

