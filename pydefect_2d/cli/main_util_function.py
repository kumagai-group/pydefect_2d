# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

from matplotlib import pyplot as plt
from pymatgen.io.vasp import Chgcar, Locpot
from vise.util.logger import get_logger

from pydefect_2d.cli.main_function import _make_gauss_charge_model, \
    _make_gauss_potential, _make_isolated_gauss, _add_z_pos
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
#    if is_sum:
#        surface_area = np.prod(vol_data.structure.lattice.lengths[:2])
#        values *= surface_area
    ax.plot(z_grid, values, color="red")
    plt.savefig(f"{args.filename}.pdf")


def make_gauss_model_from_z(args):
    """depends on the supercell size and defect position"""
    lat = args.supercell_info.structure.lattice
    grids = Grids.from_z_grid(lat.matrix[:2, :2], args.diele_dist.dist.grid)
    for z_pos in args.z_pos:
        logger.info(f"At z={z_pos}...")
        filename = _add_z_pos("gauss_charge_model.json", z_pos)
        if (args.correction_dir / filename).exists():
            logger.info(f"{filename} already exists, so skip.")
            continue

        gauss_charge = _make_gauss_charge_model(grids, args.std_dev, z_pos,
                                                args.correction_dir)

        _make_gauss_potential(args.diele_dist, gauss_charge, args.multiprocess,
                              args.correction_dir)

        _make_isolated_gauss(args.diele_dist, gauss_charge, args.k_max,
                             args.k_mesh_dist, args.correction_dir)


def make_gaussian_energies(args):
    """depends on the supercell size and defect position"""
    lat = args.supercell_info.structure.lattice
    grids = Grids.from_z_grid(lat.matrix[:2, :2], args.diele_dist.dist.grid)
    for z_pos in args.z_pos:
        logger.info(f"At z={z_pos}...")
        filename = _add_z_pos("gauss_charge_model.json", z_pos)
        if (args.correction_dir / filename).exists():
            logger.info(f"{filename} already exists, so skip.")
            continue

        gauss_charge = _make_gauss_charge_model(grids, args.std_dev, z_pos,
                                                args.correction_dir)

        _make_gauss_potential(args.diele_dist, gauss_charge, args.multiprocess,
                              args.correction_dir)

        _make_isolated_gauss(args.diele_dist, gauss_charge, args.k_max,
                             args.k_mesh_dist, args.correction_dir)
