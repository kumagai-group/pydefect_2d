# -*- coding: utf-8 -*-
#  Copyright (c) 2020 Kumagai group.
import sys

from matplotlib import pyplot as plt
from monty.serialization import loadfn


def main():
    ax = plt.gca()
    filename = sys.argv[1]
    obj = loadfn(filename)
    try:
        obj.to_plot(ax)
        pdf_name = filename.split(".")[0] + ".pdf"
        plt.savefig(pdf_name)
    except AttributeError:
        print(f"to_plot method is not implemented in {obj.__class__}")


if __name__ == "__main__":
    main()
