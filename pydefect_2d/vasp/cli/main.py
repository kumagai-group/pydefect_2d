# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys

from monty.serialization import loadfn
from pydefect.analyzer.unitcell import Unitcell
from pymatgen.core import Structure
from pymatgen.io.vasp import Locpot

from pydefect_2d.vasp.cli.main_function import plot_volumetric_data, \
    make_epsilon_distributions, make_gauss_charge_models, make_fp_1d_potential, \
    calc_potential


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for creating and analyzing VASP input and output 
    files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    # -- Plot volumetric data. --------------------------------------------
    parser_plot_volumetric_data = subparsers.add_parser(
        name="plot_volumetric_data",
        description="Plot volumetric data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['pvd'])
    parser_plot_volumetric_data.add_argument(
        "-f", "--filename", required=True, type=str,
        help="filename.")
    parser_plot_volumetric_data.set_defaults(func=plot_volumetric_data)

    # -- Make epsilon distributions. -------------------------------------------
    parser_make_epsilon_dist = subparsers.add_parser(
        name="make_epsilon_distributions",
        description="Make epsilon distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['med'])

    parser_make_epsilon_dist.add_argument(
        "-u", "--unitcell", required=True, type=Unitcell.from_yaml,
        help="unitcell.yaml file.")
    parser_make_epsilon_dist.add_argument(
        "-s", "--structure", required=True, type=Structure.from_file,
        help="POSCAR file of the 2D unitcell.")
    parser_make_epsilon_dist.add_argument(
        "-p", "--position", required=True, type=float,
        help="Position of layer in fractional coordinates.")
    parser_make_epsilon_dist.add_argument(
        "-n", "--num_grid", required=True, type=int,
        help="Number of all_grid_points.")
    parser_make_epsilon_dist.add_argument(
        "-m", "--muls", type=int, default=[1], nargs="+",
        help="Multipliers of the supercell.")
    parser_make_epsilon_dist.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing.")
    parser_make_epsilon_dist.set_defaults(func=make_epsilon_distributions)

    # -- Make gauss charge models. ---------------------------------------------
    parser_make_gauss_charge_model = subparsers.add_parser(
        name="make_gauss_charge_models",
        description="Make Gauss charge models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['mgcm'])

    parser_make_gauss_charge_model.add_argument(
        "-d", "--defect_entry", required=True, type=loadfn,
        help="defect_entry.json file.")
    parser_make_gauss_charge_model.add_argument(
        "-e", "--epsilon_dist", required=True, type=loadfn,
        help="epsilon_distribution.json file")
    parser_make_gauss_charge_model.add_argument(
        "-m", "--muls", type=int, default=[1], nargs="+",
        help="Multipliers of the supercell.")
    parser_make_gauss_charge_model.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing.")
    parser_make_gauss_charge_model.set_defaults(
        func=make_gauss_charge_models)

    # -- Calc potential. ---------------------------------------------
    parser_calc_potential = subparsers.add_parser(
        name="calc_potential",
        description="calc potential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['cp'])

    parser_calc_potential.add_argument(
        "-e", "--epsilon_dist", required=True, type=loadfn,
        help="epsilon_distribution.json file")
    parser_calc_potential.add_argument(
        "-g", "--gauss_model", required=True, type=loadfn,
        help="gauss_charge_model.json file")
    parser_calc_potential.add_argument(
        "--no_multiprocess", dest="multiprocess", action="store_false",
        help="Switch of the multiprocess.")
    parser_calc_potential.set_defaults(func=calc_potential)

    # -- Make fp 1D potential. ---------------------------------------------
    parser_make_fp_1d_potential = subparsers.add_parser(
        name="make_fp_1d_potential",
        description="Make first-principles one-dimensional potential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['mfp'])

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

    # ------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


