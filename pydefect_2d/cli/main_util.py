# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys
from pathlib import Path

from pydefect.cli.main import add_sub_parser

from pydefect_2d.cli.main import add_2d_sub_parser
from pydefect_2d.cli.main_util_function import plot_volumetric_data, \
    make_gauss_model, make_slab_model, add_vacuum, repeat_diele_dist


def parse_args_main_util_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for analyzing VASP output files for pydefect_2d.""")

    pcr_parser = add_sub_parser(argparse, name="perfect_calc_results")
    subparsers = parser.add_subparsers()
    dir_parser = add_sub_parser(argparse, name="dirs")

    dielectric_dist = add_2d_sub_parser(argparse, "diele_dist")
    gauss_charge_std_dev = add_2d_sub_parser(argparse, "gauss_charge_std_dev")
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
    parser_plot_volumetric_data.add_argument(
        "-y", "--y_range", type=float, nargs=2,
        help=".")
    parser_plot_volumetric_data.add_argument(
        "-t", "--target_val", type=float,
        help=".")
    parser_plot_volumetric_data.add_argument(
        "-z", "--z_guess", type=float, nargs="+",
        help=".")
    parser_plot_volumetric_data.set_defaults(func=plot_volumetric_data)

    # --------------------------------------------------------------------------
    parser_gauss_model = subparsers.add_parser(
        name="gauss_model",
        description=f"Make Gauss model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_std_dev, isolated_gauss,
                 corr_dir, dir_parser, no_multiprocess],
        aliases=['gm'])

    parser_gauss_model.set_defaults(func=make_gauss_model)

    # --------------------------------------------------------------------------
    parser_make_slab_model = subparsers.add_parser(
        name="slab_model",
        description="Make slab_model.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, pcr_parser, corr_dir, dir_parser],
        aliases=['sm'])

    parser_make_slab_model.set_defaults(func=make_slab_model)

    # --------------------------------------------------------------------------
    parser_add_vacuum = subparsers.add_parser(
        name="add_vacuum",
        description="Add vacuum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['av'])
    parser_add_vacuum.add_argument(
        "-l", "--length", type=float,
        help="Length of the vertical direction.")

    parser_add_vacuum.set_defaults(func=add_vacuum)

    # --------------------------------------------------------------------------
    parser_repeat_diele = subparsers.add_parser(
        name="repeat_diele",
        description="Repeat DielectricConstDist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['rd'])
    parser_repeat_diele.add_argument(
        "-m", "--mul", type=int,
        help="Repetition.")

    parser_repeat_diele.set_defaults(func=repeat_diele_dist)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_util_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


