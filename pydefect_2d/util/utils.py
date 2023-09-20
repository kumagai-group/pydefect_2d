# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from typing import List

import numpy as np
from scipy.interpolate import UnivariateSpline, interpolate
from scipy.optimize import root


def add_z_to_filename(filename: str, z: float):
    """z is in frac"""
    x, y = filename.split(".")
    return f"{x}_{z:.3f}.{y}"


def get_z_from_filename(filename) -> float:
    return float(filename.split(".json")[0].split("_")[-1])


def show_x_values(xs: np.ndarray, ys: np.ndarray, given_y: float, x_guess):
    f = interpolate.interp1d(xs, ys)

    def find_x_for_given_y(x, y_target):
        return f(x) - y_target

    solution = root(find_x_for_given_y, x0=x_guess, args=(given_y,))
    return solution.x


def with_end_point(array: np.ndarray):
    return np.append(array, array[0])
