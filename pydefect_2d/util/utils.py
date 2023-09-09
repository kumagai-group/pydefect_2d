# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

def add_z_to_filename(filename: str, z: float):
    """z is in frac"""
    x, y = filename.split(".")
    return f"{x}_{z:.3f}.{y}"


def get_z_from_filename(filename) -> float:
    return float(filename.split(".json")[0].split("_")[-1])
