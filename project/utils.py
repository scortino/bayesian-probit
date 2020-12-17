import matplotlib.pyplot as plt


def trace_plot(mc, path=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(f"Trace Plot for {len(mc)} iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value Drawn")
    ax1.plot(range(len(mc)), mc)
    ax1.legend([f"beta {i}" for i in range(len(mc[0]))], loc="lower right")
    if path is not None:
        fig.savefig(path)
    return fig
