# -*- coding: utf-8 -*-
#  Copyright (c) 2020 Kumagai group.
import sys

from matplotlib import pyplot as plt
from monty.serialization import loadfn


def main():
    filenames = sys.argv[1:]
    for filename in filenames:
        obj = loadfn(filename)
        try:
            plot(filename, obj)
        except AttributeError:
            print(f"to_plot method is not implemented in {obj.__class__}")


def plot(filename, obj):
    obj.to_plot(plt.gca())
    pdf_name = filename.replace(".json", ".pdf")
    plt.savefig(pdf_name)
    plt.clf()


if __name__ == "__main__":
    main()
