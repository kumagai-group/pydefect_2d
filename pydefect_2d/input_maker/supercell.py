# -*- coding: utf-8 -*-
#  Copyright (c) 2020. Distributed under the terms of the MIT License.

from copy import deepcopy
from functools import reduce
from itertools import product
from typing import List, Tuple

import numpy as np
from numpy.linalg import det
from pydefect.util.error_classes import SupercellError
from pymatgen.core import IStructure


class Supercell:
    def __init__(self, input_structure: IStructure, matrix: List[List[int]]):
        self.matrix = matrix
        mul_matrix = np.eye(3)
        mul_matrix[0:2, 0:2] = matrix
        self.structure = input_structure * mul_matrix

    @property
    def isotropy(self):
        lengths = self.structure.lattice.lengths
        average = np.average(lengths[:2])
        return sum([abs(length - average) for length in lengths]) / 2 / average

    # @property
    # def average_angle(self):
    #     return sum(self.structure.lattice.angles) / 3


