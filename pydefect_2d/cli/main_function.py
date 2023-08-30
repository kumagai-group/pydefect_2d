# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import glob
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from monty.serialization import loadfn
from pydefect.analyzer.defect_structure_info import DefectStructureInfo
from pydefect.cli.main_tools import parse_dirs
from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter
from pydefect.input_maker.defect_entry import DefectEntry
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_plot_json import plot
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.potential.calc_one_d_potential import Calc1DPotential, \
    OneDGaussChargeModel
from pydefect_2d.dielectric.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.dielectric.distribution import GaussianDist, StepDist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.correction.make_site_potential import make_potential_sites
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


def make_step_dielectric_distribution(args):
    def dist(grid, center, args_):
        return StepDist.from_grid(
            grid, center, args_.width, args.error_func_width)
    make_diele_dist(dist, args)


def _add_z_pos(filename: str,
               model: Union[GaussChargeModel, OneDGaussChargeModel]):
    x, y = filename.split(".")
    return f"{x}_{model.gauss_pos_in_frac:.3f}.{y}"


make_gauss_charge_model_msg = \
    """defect_structure_info.json or a set of (supercell_info.json, defect_pos) 
need to be specified."""


def make_1d_gauss_models(args):
    left, right = sorted(args.range)
    n_grid = round((right - left) / args.num_mesh) + 1
    gauss_pos = np.linspace(left, right, n_grid, endpoint=True)
    supercell = args.supercell_info.structure

    for pos in gauss_pos:
        filename = _add_z_pos("one_d_gauss_charge_model.json", charge_model)
        if Path(filename).exists():
            logger.info(f"Because {filename} exists, so skip.")
            continue

        charge_model = OneDGaussChargeModel(grid=args.diele_dist.dist.grid,
                                            sigma=args.sigma,
                                            surface=_xy_area(supercell),
                                            gauss_pos_in_frac=pos)
        calc_1d_pot = Calc1DPotential(args.diele_dist, charge_model)
        logger.info(f"{filename} is being created.")
        calc_1d_pot.potential.to_json_file(filename)
        calc_1d_pot.potential.to_plot(plt.gca())

    plt.savefig("1d_pot.pdf")


def _xy_area(structure):
    a, b = structure.lattice.matrix[:2, :2]
    return np.linalg.norm(np.cross(a, b))


def make_fp_1d_potential(args):

    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=args.axis)

    gauss_1d_pots = []

    for gauss_1d_pot in glob.glob(f'{args.gauss_1d_pot_dir}/gauss1_d_potential*json'):
        gauss_1d_pots.append(loadfn(gauss_1d_pot))

    def _inner(_dir: Path):
        locpot = Locpot.from_file(_dir / "LOCPOT")
        defect_entry: DefectEntry = loadfn(_dir / "defect_entry.json")
        length = defect_entry.structure.lattice.lengths[args.axis]
        grid_num = locpot.dim[args.axis]
        grid = Grid(length, grid_num)
        defect_pot = locpot.get_average_along_axis(ind=args.axis)
        try:
            # "-" is needed because the VASP potential is defined for electrons.
            pot = (-(defect_pot - perfect_pot)).tolist()
        except ValueError:
            print("The size of two LOCPOT files seems different.")
            raise

        fp_potential = Fp1DPotential(grid, pot)

        grads, gaussian_pos = [], []

        for gauss_1d_pot in gauss_1d_pots:
            gauss_pot = deepcopy(gauss_1d_pot)
            gauss_pot.potential *= defect_entry.charge
            diff = OneDPotDiff(fp_pot=fp_potential, gauss_pot=gauss_pot)
            grads.append(diff.potential_diff_gradient)
            gaussian_pos.append(gauss_1d_pot.gauss_positions)

        pot_grads = PotDiffGradients(grads, gaussian_pos)
        pot_grads.to_json_file()
        pot_grads.to_plot(plt.gca())
        plt.show()

        fp_potential.gauss_positions = pot_grads.gauss_pos_w_min_grad()
        print(fp_potential.gauss_positions)
        fp_potential.to_json_file()

    file_name = "fp1_d_potential.json"
    parse_dirs(args.dirs, _inner, args.verbose, file_name)


def make_gauss_charge_model(args):
    """depends on the supercell size and defect position"""
    try:
        if args.defect_structure_info:
            dsi: DefectStructureInfo = args.defect_structure_info
            lat = dsi.shifted_final_structure.lattice
            defect_z_pos = dsi.center[2]
        else:
            lat = args.supercell_info.structure.lattice
            defect_z_pos = float(args.defect_z_pos)
    except (AssertionError, TypeError):
        logger.info(make_gauss_charge_model_msg)
        raise

    grids = Grids.from_z_grid(lat.matrix[:2, :2],
                              args.diele_dist.dist.grid)

    model = GaussChargeModel(grids, args.sigma, defect_z_pos)
    filename = _add_z_pos(model.json_filename, model)
    model.to_json_file(filename)
    return model


def calc_gauss_charge_potential(args):
    """depends on the supercell size and defect position"""
    potential = CalcGaussChargePotential(
        dielectric_const=args.diele_dist,
        gauss_charge_model=args.gauss_charge_model,
        multiprocess=args.multiprocess).potential
    filename = _add_z_pos(potential.json_filename, args.gauss_charge_model)
    potential.to_json_file(filename)
    return potential


def make_isolated_gauss_energy(args):
    """depends on the supercell size, defect position"""
    # static = args.diele_dist.static
    # try:
    #     assert_almost_equal(diele[0], diele[1])
    # except AssertionError:
    #     logger.info("Only the case where static dielectric constant is "
    #                 "isotropic in xy-plane.")
    #     raise

    isolated = IsolatedGaussEnergy(gauss_charge_model=args.gauss_charge_model,
                                   diele_const_dist=args.diele_dist,
                                   k_max=args.k_max,
                                   num_k_mesh=args.num_k_mesh)
    print(isolated)
    filename = _add_z_pos(isolated.json_filename, args.gauss_charge_model)
    isolated.to_json_file(filename)
    return isolated


def _get_obj(dir_: Path, filename: str, fp_potential: Fp1DPotential):
    x, y = filename.split(".")
    if fp_potential:
        filename = dir_ / f"{x}_{fp_potential.gauss_pos:.3f}.{y}"
    else:
        filename = dir_ / f"{x}.{y}"
    try:
        return loadfn(filename)
    except FileNotFoundError:
        print(f"{filename} is not found.")
        raise


def make_slab_model(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    d, fp = args.correction_dir, args.fp_potential
    gauss_charge_model = _get_obj(d, "gauss_charge_model.json", fp)
    gauss_charge_pot = _get_obj(d, "gauss_charge_potential.json", fp)

    charge = fp.charge_state if fp else 1
    slab_model = SlabModel(diele_dist=args.diele_dist,
                           gauss_charge_model=gauss_charge_model,
                           gauss_charge_potential=gauss_charge_pot,
                           charge_state=charge,
                           fp_potential=fp)
    slab_model.to_json_file()
    ProfilePlotter(plt, slab_model)
    plt.savefig("potential_profile.pdf")


def make_correction(args):
    """depends on the supercell size, defect position and charge

    This should be placed at each defect calc dir.
    """
    fp_potential = loadfn(args.dir / "fp1_d_potential.json")
    calc_results = loadfn(args.dir / "calc_results.json")
    slab_model = loadfn(args.dir / "slab_model.json")

    isolated_gauss_energy = _get_obj(args.correction_dir,
                                     "isolated_gauss_energy.json",
                                     fp_potential)
    squared_charge_state = slab_model.charge_state ** 2
    isolated_energy = isolated_gauss_energy.self_energy * squared_charge_state
    # correction = Gauss2dCorrection(slab_model.charge_state,
    #                                slab_model.electrostatic_energy,
    #                                isolated_energy,
    #                                slab_model.potential_diff)
    # print(correction)
    # correction.to_json_file()

    sites = make_potential_sites(calc_results,
                                 args.perfect_calc_results,
                                 slab_model)
    plotter = SitePotentialMplPlotter(
        title="atomic site potential", sites=sites)
    plotter.construct_plot()
    plotter.plt.savefig(fname="atomic_site_potential.pdf")
    plotter.plt.clf()


def make_corr(args):
    d, fp = args.correction_dir, args.fp_potential
    args.diele_dist = loadfn(d / "dielectric_const_dist.json")
    args.defect_structure_info = None
    args.defect_z_pos = args.fp_potential.gauss_positions

    try:
        gauss_charge_model = _get_obj(d, "gauss_charge_model.json", fp)
    except FileNotFoundError:
        gauss_charge_model = make_gauss_charge_model(args)
        for f in glob.glob("gauss_charge_model*.json"):
            shutil.move(f, d)

    args.gauss_charge_model = gauss_charge_model

    try:
        gauss_charge_pot = _get_obj(d, "gauss_charge_potential.json", fp)
    except FileNotFoundError:
        gauss_charge_pot = calc_gauss_charge_potential(args)
        for f in glob.glob("gauss_charge_potential*.json"):
            shutil.move(f, d)

    try:
        isolated_gauss_energy = _get_obj(d, "isolated_gauss_energy.json", fp)
    except FileNotFoundError:
        isolated_gauss_energy = make_isolated_gauss_energy(args)
        for f in glob.glob("isolated_gauss_energy*.json"):
            shutil.move(f, d)

    try:
        slab_model = _get_obj(d, "slab_model.json", fp)
    except FileNotFoundError:
        slab_model = make_slab_model(args)
        for f in glob.glob("slab_model.json"):
            shutil.move(f, d)

    args.slab_model = slab_model
    make_correction(args)