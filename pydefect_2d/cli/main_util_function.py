# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from matplotlib import pyplot as plt
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_function import _make_gauss_charge_model, \
    _make_gauss_potential, _make_isolated_gauss
from pydefect_2d.util.utils import add_z_to_filename, show_x_values
from pydefect_2d.correction.gauss_energy import make_gauss_energies
from pydefect_2d.potential.grids import Grids

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


