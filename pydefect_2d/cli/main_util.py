# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys

from pydefect.cli.main import add_sub_parser

from pydefect_2d.cli.main import add_2d_sub_parser
from pydefect_2d.cli.main_util_function import plot_volumetric_data, \
    make_gauss_model_from_z, make_gaussian_energies_from_args


def parse_args_main_util_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for analyzing VASP output files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    supercell_info_parser = add_sub_parser(argparse, name="supercell_info")

    dielectric_dist = add_2d_sub_parser(argparse, "diele_dist")
    gauss_charge_std_dev = add_2d_sub_parser(argparse, "std_dev")
    isolated_gauss = add_2d_sub_parser(argparse, "isolated_gauss")
    corr_dir = add_2d_sub_parser(argparse, "corr_dir")
    no_multiprocess = add_2d_sub_parser(argparse, "no_multiprocess")

    # --------------------------------------------------------------------------
    parser_plot_volumetric_data = subparsers.add_parser(
        name="plot_volumetric_data",
        description="Plot volumetric data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['pvd'])
    parser_plot_volumetric_data.add_argument(
        "-f", "--filename", required=True, type=str,
        help="Filename.")
    parser_plot_volumetric_data.add_argument(
        "-d", "--direction", type=int, default=2,
        help="Plot direction.")
    parser_plot_volumetric_data.set_defaults(func=plot_volumetric_data)

    # --------------------------------------------------------------------------
    parser_gauss_model = subparsers.add_parser(
        name="gauss_model_from_z",
        description=f"Make Gauss models at given z sites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[supercell_info_parser, dielectric_dist, gauss_charge_std_dev,
                 isolated_gauss, corr_dir, no_multiprocess],
        aliases=['gmz'])

    parser_gauss_model.add_argument(
        "-z", "--z_pos", type=float, nargs="+", required=True,
        help="Positions gauss models along z in frac coords.")
    parser_gauss_model.set_defaults(func=make_gauss_model_from_z)

    # --------------------------------------------------------------------------
    parser_gauss_energies = subparsers.add_parser(
        name="gauss_energies",
        description=f"Make Gaussian energies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[corr_dir],
        aliases=['ge'])

    parser_gauss_energies.add_argument(
        "-z", "--z_range", type=float, nargs=2, help="Z range")
    parser_gauss_energies.set_defaults(func=make_gaussian_energies_from_args)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_util_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


