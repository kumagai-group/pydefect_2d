# -*- coding: utf-8 -*-
#  Copyright (c) 2020 Kumagai group.
import sys
from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from monty.serialization import loadfn


def main():
    filenames = sys.argv[1:]
    plot(filenames, plt.gca())
    if len(filenames) == 1:
        pdf_name = filenames[0].replace(".json", ".pdf")
    else:
        pdf_name = "plot.pdf"
    plt.savefig(pdf_name)
    plt.clf()


# def plot(filename, obj):
#     obj.to_plot(plt.gca())


def plot(files: List[str], ax: Axes):
    for file in files:
        obj = loadfn(file)
        try:
            obj.to_plot(ax)
        except AttributeError:
            print(f"to_plot method is not implemented in {obj.__class__}")
            continue


if __name__ == "__main__":
    main()
