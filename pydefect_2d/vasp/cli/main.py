# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys

from pydefect.analyzer.unitcell import Unitcell
from pymatgen.core import Structure

from pydefect_2d.vasp.cli.main_function import make_epsilon_distribution


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for creating and analyzing VASP input and output 
    files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    # -- Make epsilon distribution. --------------------------------------------
    parser_make_epsilon_distribution = subparsers.add_parser(
        name="make_epsilon_distribution",
        description="Make epsilon distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['med'])

    parser_make_epsilon_distribution.add_argument(
        "-u", "--unitcell", required=True, type=Unitcell.from_yaml,
        help="unitcell.yaml file.")
    parser_make_epsilon_distribution.add_argument(
        "-s", "--structure", required=True, type=Structure.from_file,
        help="POSCAR file of the 2D unitcell.")
    parser_make_epsilon_distribution.add_argument(
        "-p", "--position", required=True, type=float,
        help="Position of layer in fractional coordinates.")
    parser_make_epsilon_distribution.add_argument(
        "-n", "--num_grid", required=True, type=int,
        help="Number of grid.")
    parser_make_epsilon_distribution.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing.")
    parser_make_epsilon_distribution.set_defaults(func=make_epsilon_distribution)

    # ------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


