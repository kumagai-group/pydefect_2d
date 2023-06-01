# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys
from pathlib import Path

from monty.serialization import loadfn
from pydefect.analyzer.unitcell import Unitcell
from pymatgen.core import Structure
from pymatgen.io.vasp import Locpot

from pydefect_2d.vasp.cli.main_function import make_epsilon_distribution, \
    make_slab_gauss_model, plot_volumetric_data


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

    # -- Make epsilon distribution. --------------------------------------------
    parser_make_epsilon_dist = subparsers.add_parser(
        name="make_epsilon_distribution",
        description="Make epsilon distribution.",
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
        help="Number of grid.")
    parser_make_epsilon_dist.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing.")
    parser_make_epsilon_dist.set_defaults(func=make_epsilon_distribution)

    # -- Make SlabGaussModel. --------------------------------------------
    parser_make_slab_gauss_model = subparsers.add_parser(
        name="make_slab_gauss_model",
        description="Make SlabGaussModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['msgm'])

    parser_make_slab_gauss_model.add_argument(
        "-d", "--defect_entry", required=True, type=loadfn,
        help="defect_entry.json file.")
    parser_make_slab_gauss_model.add_argument(
        "-dl", "--defect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a defect calculation.")
    parser_make_slab_gauss_model.add_argument(
        "-pl", "--perfect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a perfect supercell calculation.")
    parser_make_slab_gauss_model.add_argument(
        "-e", "--epsilon_dist", required=True, type=loadfn,
        help="epsilon_distribution.json file")
    parser_make_slab_gauss_model.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing.")
    parser_make_slab_gauss_model.add_argument(
        "--no_potential_calc", dest="calc_potential", action="store_false",
        help="Set if potential needs not to be calcualted.")
    parser_make_slab_gauss_model.add_argument(
        "--grid_divisor", dest="grid_divisor", type=int, default=10,
        help=".")
    parser_make_slab_gauss_model.set_defaults(func=make_slab_gauss_model)

    # ------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


