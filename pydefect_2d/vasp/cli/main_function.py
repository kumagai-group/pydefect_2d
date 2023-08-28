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

from pydefect_2d.correction.correction_2d import Gauss2dCorrection
from pydefect_2d.correction.isolated_gauss import IsolatedGaussEnergy
from pydefect_2d.potential.calc_one_d_potential import Calc1DPotential, \
    OneDGaussChargeModel
from pydefect_2d.potential.dielectric_distribution import \
    DielectricConstDist
from pydefect_2d.potential.distribution import GaussianDist, StepDist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.make_site_potential import make_potential_sites
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


def make_dielectric_distribution(args):
    """depends on the supercell size"""
    electronic = list(np.diag(args.unitcell.ele_dielectric_const))
    ionic = list(np.diag(args.unitcell.ion_dielectric_const))
    grid = Grid(args.structure.lattice.c, args.num_grid)

    if args.type == "gauss":
        position = args.structure.lattice.c * args.position
        dist = GaussianDist.from_grid(grid, position, args.sigma)
    elif args.type == "step":
        left = args.structure.lattice.c * args.step_left
        right = args.structure.lattice.c * args.step_right
        dist = StepDist.from_grid(grid, left, right, args.error_func_width)
    else:
        raise ValueError

    diele = DielectricConstDist(electronic, ionic, dist)
    diele.to_json_file()


def _add_z_pos(filename: str,
               model: Union[GaussChargeModel, OneDGaussChargeModel]):
    x, y = filename.split(".")
    return f"{x}_{model.gauss_pos_in_frac:.3f}.{y}"


make_gauss_charge_model_msg = \
    """defect_structure_info.json or a set of (supercell_info.json, defect_pos) 
need to be specified."""


def make_1d_gauss_models(args):
    left, right = args.range
    assert left < right
    assert right - left < 0.5
    gauss_pos = np.linspace(left, right, args.num_mesh, endpoint=True)

    supercell: Structure = args.supercell_info.structure
    a, b = supercell.lattice.matrix[:2, :2]
    c = supercell.lattice.c
    xy_area = np.linalg.norm(np.cross(a, b))

    diele = args.dielectric_dist
    diele.dist.num_grid = args.num_grid
    grid = Grid(c, args.num_grid)

    for pos in gauss_pos:
        charge_model = OneDGaussChargeModel(grid=grid,
                                            sigma=args.sigma,
                                            surface=xy_area,
                                            gauss_pos_in_frac=pos)
        calc_1d_pot = Calc1DPotential(diele, charge_model,
                                      effective=args.effective)

        pot = calc_1d_pot.potential
        filename = _add_z_pos(pot.json_filename, charge_model)
        pot.to_json_file(filename)

        pot.to_plot(plt.gca())
    plt.savefig("1d_pot.pdf")

#        SlabModel(args.dielectric_dist, charge_model, pot, charge_state=1)


def set_gauss_pos(args):
    grads, gaussian_pos = [], []

    for gauss_1d_pot in args.gauss_1d_pots:
        gauss_1d_pot.charge_state = args.defect_entry.charge
        diff = OneDPotDiff(fp_pot=args.fp_potential, gauss_pot=gauss_1d_pot)
        grads.append(diff.potential_diff_gradient)
        gaussian_pos.append(gauss_1d_pot.gauss_positions)

    pot_grads = PotDiffGradients(grads, gaussian_pos)
    pot_grads.to_json_file()
    pot_grads.to_plot(plt.gca())
    plt.show()

    args.fp_potential.gauss_positions = pot_grads.gauss_pos_w_min_grad()
    print(args.fp_potential.gauss_positions)
    args.fp_potential.to_json_file()


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
                              args.dielectric_dist.dist.grid)

    model = GaussChargeModel(grids, args.sigma, defect_z_pos)
    filename = _add_z_pos(model.json_filename, model)
    model.to_json_file(filename)
    return model


def calc_gauss_charge_potential(args):
    """depends on the supercell size and defect position"""
    potential = CalcGaussChargePotential(
        dielectric_const=args.dielectric_dist,
        gauss_charge_model=args.gauss_charge_model,
        multiprocess=args.multiprocess,
        effective=args.effective).potential
    filename = _add_z_pos(potential.json_filename, args.gauss_charge_model)
    potential.to_json_file(filename)
    return potential


def make_isolated_gauss_energy(args):
    """depends on the supercell size, defect position"""
    # static = args.dielectric_dist.static
    # try:
    #     assert_almost_equal(diele[0], diele[1])
    # except AssertionError:
    #     logger.info("Only the case where static dielectric constant is "
    #                 "isotropic in xy-plane.")
    #     raise

    isolated = IsolatedGaussEnergy(gauss_charge_model=args.gauss_charge_model,
                                   diele_const_dist=args.dielectric_dist,
                                   k_max=args.k_max,
                                   num_k_mesh=args.num_k_mesh,
                                   effective=False)
    print(isolated)
    filename = _add_z_pos(isolated.json_filename, args.gauss_charge_model)
    isolated.to_json_file(filename)
    return isolated


def make_fp_1d_potential(args):
    length = args.defect_locpot.structure.lattice.lengths[args.axis]
    grid_num = args.defect_locpot.dim[args.axis]
    grid = Grid(length, grid_num)
    if args.defect_entry:
        charge = args.defect_entry.charge
    else:
        charge = args.charge

    defect_pot = args.defect_locpot.get_average_along_axis(ind=args.axis)
    perfect_pot = args.perfect_locpot.get_average_along_axis(ind=args.axis)

    try:
        # "-" is needed because the VASP potential is defined for electrons.
        pot = (-(defect_pot - perfect_pot)).tolist()
    except ValueError:
        print("The size of two LOCPOT files seems different.")
        raise

    Fp1DPotential(grid, pot, charge).to_json_file()


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
    slab_model = SlabModel(diele_dist=args.dielectric_dist,
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
    args.dielectric_dist = loadfn(d / "dielectric_const_dist.json")
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