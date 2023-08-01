# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass

from pydefect.corrections.abstract_correction import Correction
from tabulate import tabulate


@dataclass
class Gauss2dCorrection(Correction):
    """
    """
    charge: int
    periodic_energy: float
    isolated_energy: float
    alignment: float

    @property
    def alignment_correction(self):
        return - self.charge * self.alignment

    @property
    def gauss_charge_term(self):
        return self.isolated_energy - self.periodic_energy

    @property
    def correction_energy(self):
        return self.gauss_charge_term + self.alignment_correction

    def __str__(self):
        table = [[f"charge:", self.charge],
                 [f"periodic energy:", f"{self.periodic_energy:.4}"],
                 [f"isolated energy:", f"{self.isolated_energy:.4}"],
                 [f"gauss charge correction:", f"{self.gauss_charge_term:.4}"],
                 [f"alignment correction:", f"{self.alignment_correction:.4}"],
                 [f"total correction:", f"{self.correction_energy:.4}"]]
        return tabulate(table, tablefmt="plain", stralign="right")

    @property
    def correction_dict(self):
        return {"gauss charge term": self.gauss_charge_term,
                "alignment term": self.alignment_correction}



