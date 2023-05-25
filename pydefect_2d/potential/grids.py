# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List

from monty.json import MSONable
from numpy import linspace
from vise.util.mix_in import ToJsonFileMixIn


@dataclass
class Grids(MSONable, ToJsonFileMixIn):
    lattice_constants: List[float]  # assume orthogonal system
    num_grids: List[int]

    @property
    def grids(self):
        return [linspace(0, lat, grids, False)
                for lat, grids in zip(self.lattice_constants, self.num_grids)]