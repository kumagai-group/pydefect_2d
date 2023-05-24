# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List


@dataclass
class ExtrapolationModel:
    lattice_constants: List[float]  # assume orthogonal system
    slab_begin_at: float  # in fractional coord.
    slab_end_at: float
    bulk_epsilon: float
    charge: float
    std_dev: float
    defect_z_pos: float  # in fractional coord. x=y=0