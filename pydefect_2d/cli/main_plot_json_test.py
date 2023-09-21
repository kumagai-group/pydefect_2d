# -*- coding: utf-8 -*-
#  Copyright (c) 2020 Kumagai group.
from matplotlib import pyplot as plt

from pydefect_2d.cli.main_plot_json import plot


def test_plot(test_files):
    d = test_files / "1d_gauss"
    ax = plt.gca()
    plot(files=[str(d / "1d_gauss_potential_0.300.json"),
                str(d / "1d_gauss_potential_0.310.json")], ax=ax)
    ax.legend()
    plt.show()