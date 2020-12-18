import os

import matplotlib.pyplot as plt
import numpy as np


def load_finney47():
    dataset = np.loadtxt("./data/finney47.csv", delimiter=",", skiprows=1)
    X = dataset[:, 1:]
    Y = dataset[:, 0]
    return X, Y

def trace_plot(mc, path=None, replace=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(f"Trace Plot for {len(mc)-1} iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value Drawn")
    ax1.plot(range(len(mc)), mc)
    ax1.legend([f"beta {i}" for i in range(len(mc[0]))], loc="lower right")
    if path is not None:
        if not os.path.exists(path) or replace:
            fig.savefig(path)
    return fig
