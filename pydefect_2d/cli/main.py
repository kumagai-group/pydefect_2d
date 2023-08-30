# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys
from pathlib import Path

from monty.serialization import loadfn
from pydefect.cli.main import add_sub_parser
from pymatgen.core import Structure
from pymatgen.io.vasp import Locpot

from pydefect_2d.cli.main_function import make_gauss_dielectric_distribution, \
    make_step_dielectric_distribution, make_gauss_charge_model, \
    make_fp_1d_potential, \
    calc_gauss_charge_potential, make_slab_model, make_isolated_gauss_energy, \
    make_correction, make_gauss_charge_model_msg, \
    make_1d_gauss_models, make_corr


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for creating and analyzing VASP input and output 
    files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    pcr_parser = add_sub_parser(argparse, name="perfect_calc_results")
    unitcell_parser = add_sub_parser(argparse, name="unitcell")
    dir_parser = add_sub_parser(argparse, name="dirs")

    gauss_charge_model = argparse.ArgumentParser(description="", add_help=False)
    gauss_charge_model.add_argument(
        "-g", "--gauss_charge_model", required=True, type=loadfn,
        help="gauss_charge_model.json file")

    dielectric_dist = argparse.ArgumentParser(description="", add_help=False)
    dielectric_dist.add_argument(
        "-d", "--dielectric_dist", required=True, type=loadfn,
        help="dielectric_distribution.json file")

    perfect_slab = argparse.ArgumentParser(description="", add_help=False)
    perfect_slab.add_argument(
        "-p", "--perfect_slab", required=True, type=Structure.from_file,
        help="POSCAR file of the perfect slab model.")

    z_num_grid = argparse.ArgumentParser(description="", add_help=False)
    z_num_grid.add_argument(
        "-n", "--num_grid", type=int,
        help="Number of grid along z direction.")

    center = argparse.ArgumentParser(description="", add_help=False)
    center.add_argument(
        "-c", "--center", type=float,
        help="Center position of layer in fractional coordinates.")

    defect_entry = argparse.ArgumentParser(description="", add_help=False)
    defect_entry.add_argument(
        "-de", "--defect_entry", type=loadfn,
        help="defect_entry.json file.")

    isolated_gauss = argparse.ArgumentParser( description="", add_help=False)
    isolated_gauss.add_argument(
        "--k_max", type=float, default=6.0,
        help="Max of k integration range.")
    isolated_gauss.add_argument(
        "--num_k_mesh", type=int, default=100,
        help="Number of mesh.")

    # --------------------------------------------------------------------------
    parser_make_gauss_dielectric_dist = subparsers.add_parser(
        name="make_gauss_dielectric_distribution",
        description="Make gauss dielectric distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, perfect_slab, z_num_grid, center],
        aliases=['gdd'])

    parser_make_gauss_dielectric_dist.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing in Å.")
    parser_make_gauss_dielectric_dist.set_defaults(
        func=make_gauss_dielectric_distribution)

    # --------------------------------------------------------------------------
    parser_make_step_dielectric_dist = subparsers.add_parser(
        name="make_step_dielectric_distributions",
        description="Make step-like dielectric distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, perfect_slab, z_num_grid, center],
        aliases=['sdd'])

    parser_make_step_dielectric_dist.add_argument(
        "-w", "--width", type=float,
        help="")
    parser_make_step_dielectric_dist.add_argument(
        "--error_func_width", type=float, default=0.3,
        help="Width of error function in Å")
    parser_make_step_dielectric_dist.set_defaults(
        func=make_step_dielectric_distribution)

    # --------------------------------------------------------------------------
    parser_make_1d_gauss_model = subparsers.add_parser(
        name="make_1d_gauss_models",
        description=f"Make 1D Gauss models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['ogm'])

    parser_make_1d_gauss_model.add_argument(
        "-r", "--range", type=float, nargs=2,
        help="Position range of gauss charge in fractional coord.")
    parser_make_1d_gauss_model.add_argument(
        "-si", "--supercell_info", type=loadfn,
        help="supercell_info.json file.")
    parser_make_1d_gauss_model.add_argument(
        "-ng", "--num_grid", type=int,
        help="Number of FFT xy_grids.")
    parser_make_1d_gauss_model.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing in Å.")
    parser_make_1d_gauss_model.add_argument(
        "-n", "--num_mesh", type=int, default=10,
        help="Number of mesh for gauss charge positions.")
    parser_make_1d_gauss_model.set_defaults(func=make_1d_gauss_models)

    # --------------------------------------------------------------------------
    parser_make_fp_1d_potential = subparsers.add_parser(
        name="make_fp_1d_potential",
        description="Make planar averaged 1D potential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dir_parser],
        aliases=['fp'])

    parser_make_fp_1d_potential.add_argument(
        "-pl", "--perfect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a perfect supercell calculation.")
    parser_make_fp_1d_potential.add_argument(
        "-a", "--axis", type=int, choices=[0, 1, 2], default=2,
        help="Set axis along the normal direction to slab model. "
             "0, 1, and 2 correspond to x, y, and z directions, respectively")
    parser_make_fp_1d_potential.add_argument(
        "-g", "--gauss_1d_pot_dir", type=Path,
        help="")
    parser_make_fp_1d_potential.set_defaults(func=make_fp_1d_potential)

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

    parser_isolated_gauss_energy.set_defaults(func=make_isolated_gauss_energy)

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
    parser_make_slab_model.set_defaults(func=make_slab_model)

    # --------------------------------------------------------------------------
    parser_make_correction = subparsers.add_parser(
        name="make_correction",
        description="Make 2d point defect correction.",
        parents=[pcr_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['c'])

    parser_make_correction.add_argument(
        "-d", "--dir", required=True, type=Path,
        help="")
    parser_make_correction.add_argument(
        "-cd", "--correction_dir", required=True, type=Path,
        help="")
    # parser_make_correction.add_argument(
    #     "-fp", "--fp_potential", type=loadfn,
    #     help="fp_potential.json file")
    # parser_make_correction.add_argument(
    #     "-s", "--slab_model", type=loadfn,
    #     help="slab_model.json file.")
    parser_make_correction.set_defaults(func=make_correction)

    # --------------------------------------------------------------------------
    parser_make_corr = subparsers.add_parser(
        name="make_corr",
        description="Make 2d point defect correction.",
        parents=[isolated_gauss],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=['mc'])

    parser_make_corr.add_argument(
        "-cd", "--correction_dir", required=True, type=Path,
        help="")
    parser_make_corr.add_argument(
        "-fp", "--fp_potential", type=loadfn,
        help="fp_potential.json file")
    parser_make_corr.add_argument(
        "-si", "--supercell_info", type=loadfn,
        help="supercell_info.json file.")
    parser_make_corr.add_argument(
        "--no_multiprocess", dest="multiprocess", action="store_false",
        help="Switch of the multiprocess.")
    parser_make_corr.add_argument(
        "--sigma", default=0.5, type=float,
        help="Sigma of the gaussian smearing in Å.")
    parser_make_corr.set_defaults(func=make_corr)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


