# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys
from pathlib import Path

from monty.serialization import loadfn
from pydefect.analyzer.unitcell import Unitcell
from pymatgen.core import Structure
from pymatgen.io.vasp import Locpot

from pydefect_2d.vasp.cli.main_function import plot_volumetric_data, \
    make_gauss_charge_model, make_fp_1d_potential, \
    calc_gauss_charge_potential, make_slab_model, make_isolated_gauss_energy, \
    make_correction, make_dielectric_distribution, make_gauss_charge_model_msg


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for creating and analyzing VASP input and output 
    files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    # -- parent parser
    gauss_charge_model = argparse.ArgumentParser(
        description="", add_help=False)
    gauss_charge_model.add_argument(
        "-g", "--gauss_charge_model", required=True, type=loadfn,
        help="single_gauss_charge_model.json file")

    # -- parent parser
    dielectric_dist = argparse.ArgumentParser(
        description="", add_help=False)
    dielectric_dist.add_argument(
        "-d", "--dielectric_dist", required=True, type=loadfn,
        help="dielectric_distribution.json file")

    # --------------------------------------------------------------------------
    parser_plot_volumetric_data = subparsers.add_parser(
        name="plot_volumetric_data",
        description="Plot volumetric data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['pvd'])
    parser_plot_volumetric_data.add_argument(
        "-f", "--filename", required=True, type=str,
        help="filename.")
    parser_plot_volumetric_data.set_defaults(func=plot_volumetric_data)

    # --------------------------------------------------------------------------
    parser_make_dielectric_dist = subparsers.add_parser(
        name="make_dielectric_distributions",
        description="Make dielectric distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['dd'])

    parser_make_dielectric_dist.add_argument(
        "-u", "--unitcell", required=True, type=Unitcell.from_yaml,
        help="unitcell.yaml file.")
    parser_make_dielectric_dist.add_argument(
        "-s", "--structure", required=True, type=Structure.from_file,
        help="POSCAR file of the 2D unitcell.")
    parser_make_dielectric_dist.add_argument(
        "-n", "--num_grid", required=True, type=int,
        help="Number of all_grid_points.")
    parser_make_dielectric_dist.add_argument(
        "-t", "--type", required=True, type=str, choices=["gauss", "step"])
    # -- gauss
    parser_make_dielectric_dist.add_argument(
        "-p", "--position", type=float,
        help="Position of layer in fractional coordinates.")
    parser_make_dielectric_dist.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing in Å.")
    # -- step
    parser_make_dielectric_dist.add_argument(
        "-sl", "--step_left", type=float,
        help="Position of left side of step function in frac. coord.")
    parser_make_dielectric_dist.add_argument(
        "-sr", "--step_right", type=float,
        help="Position of right side of step function in frac. coord.")
    parser_make_dielectric_dist.add_argument(
        "--error_func_width", type=float, default=0.3,
        help="Width of error function in Å")
    parser_make_dielectric_dist.set_defaults(func=make_dielectric_distribution)

    # --------------------------------------------------------------------------
    parser_make_gauss_charge_model = subparsers.add_parser(
        name="make_gauss_charge_model",
        description=f"Make Gauss charge models. {make_gauss_charge_model_msg}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['gcm'])

    parser_make_gauss_charge_model.add_argument(
        "-dsi", "--defect_structure_info", type=loadfn,
        help="defect_structure_info.json file.")
    parser_make_gauss_charge_model.add_argument(
        "-si", "--supercell_info", type=loadfn,
        help="supercell_info.json file.")
    parser_make_gauss_charge_model.add_argument(
        "-dp", "--defect_z_pos", type=float,
        help="Defect position along z direction in fractional coord.")
    parser_make_gauss_charge_model.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing in Å.")
    parser_make_gauss_charge_model.set_defaults(
        func=make_gauss_charge_model)

    # --------------------------------------------------------------------------
    parser_calc_potential = subparsers.add_parser(
        name="calc_gauss_charge_potential",
        description="calc potential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_model],
        aliases=['gcp'])

    parser_calc_potential.add_argument(
        "--no_multiprocess", dest="multiprocess", action="store_false",
        help="Switch of the multiprocess.")
    parser_calc_potential.set_defaults(func=calc_gauss_charge_potential)

    # --------------------------------------------------------------------------
    parser_isolated_gauss_energy = subparsers.add_parser(
        name="make_isolated_gauss_energy",
        description="Calculate the isolated gauss energy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_model],
        aliases=['ige'])

    parser_isolated_gauss_energy.add_argument(
        "--k_max", type=float, default=6.0,
        help="Max of k integration range.")
    parser_isolated_gauss_energy.add_argument(
        "--k_mesh_dist", type=float, default=0.1,
        help="Mesh distance of k integration.")
    parser_isolated_gauss_energy.set_defaults(func=make_isolated_gauss_energy)

    # --------------------------------------------------------------------------
    parser_make_fp_1d_potential = subparsers.add_parser(
        name="make_fp_1d_potential",
        description="Make planar averaged 1D potential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['fp'])

    parser_make_fp_1d_potential.add_argument(
        "-dl", "--defect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a defect calculation.")
    parser_make_fp_1d_potential.add_argument(
        "-pl", "--perfect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a perfect supercell calculation.")
    parser_make_fp_1d_potential.add_argument(
        "-a", "--axis", type=int, choices=[0, 1, 2], default=2,
        help="Set axis along the normal direction to slab model. "
             "0, 1, and 2 correspond to x, y, and z directions, respectively")
    parser_make_fp_1d_potential.set_defaults(func=make_fp_1d_potential)

    # --------------------------------------------------------------------------
    parser_make_slab_model = subparsers.add_parser(
        name="make_slab_model",
        description="Make slab_model.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['sm'])

    parser_make_slab_model.add_argument(
        "-cd", "--correction_dir", required=True, type=Path,
        help="")
    parser_make_slab_model.add_argument(
        "-fp", "--fp_potential", type=loadfn,
        help="fp_potential.json file")
    parser_make_slab_model.add_argument(
        "-de", "--defect_entry", required=True, type=loadfn,
        help="defect_entry.json file.")
    parser_make_slab_model.set_defaults(func=make_slab_model)

    # --------------------------------------------------------------------------
    parser_make_correction = subparsers.add_parser(
        name="make_correction",
        description="Make 2d point defect correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['c'])

    parser_make_correction.add_argument(
        "-de", "--defect_entry", required=True, type=loadfn,
        help="defect_entry.json file.")
    parser_make_correction.add_argument(
        "-cd", "--correction_dir", required=True, type=Path,
        help="")
    parser_make_correction.add_argument(
        "-s", "--slab_model", type=loadfn,
        help="slab_model.json file.")
    parser_make_correction.set_defaults(func=make_correction)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


