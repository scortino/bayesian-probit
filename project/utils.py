import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_finney47():
    dataset = np.loadtxt("./data/finney47.csv", delimiter=",", skiprows=1)
    X = dataset[:, 1:]
    Y = dataset[:, 0]
    return X, Y

def trace_plot(mc, path=None, replace=False, title_prefix=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(f"{title_prefix}Trace Plot for {len(mc)-1} iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value Drawn")
    ax1.plot(range(len(mc)-1), mc[1:])
    ax1.legend([f"beta {i}" for i in range(len(mc[0]))], loc="lower right")
    if path is not None:
        if not os.path.exists(path) or replace:
            fig.savefig(path)
    return fig

def dist_plot(mc, warmup=200, path=None, replace=False, title_prefix=""):
    mc = np.vstack(mc) # T, k
    k = mc.shape[1]
    fig, axs = plt.subplots(k, squeeze=True)
    fig.suptitle(f"{title_prefix}Distribution Plot for Beta after {len(mc)-1} iterations")
    for i, ax in enumerate(axs):
        sns.kdeplot(mc[warmup+1:, i], ax=ax)
        ax.set_xlabel(f"beta_{i}")
        ax.set_ylabel("Density")
    if path is not None:
        if not os.path.exists(path) or replace:
            fig.savefig(path)
    return fig
