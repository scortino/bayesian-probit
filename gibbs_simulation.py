from pathlib import Path

from project.gibbs import GibbsProbit
from project.utils import load_finney47, trace_plot, dist_plot


METHOD = 'gibbs'
BASE_PATH = Path('./images')
SEED = 42

# Finney (1947), see paper p. 675
X, Y = load_finney47()

# Function that, given n_iter and warmup, runs the algorithm on the data for three different priors
def simul(n_iter, warmup):
    PRIOR = "noinfo"
    m = GibbsProbit(prior="non-informative", intercept=True)
    mc = m.fit(X, Y, n_iter=n_iter, warmup=warmup, seed=SEED)
    trace_plot(mc, path=BASE_PATH/f"trace_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=False, title_prefix=f"{METHOD.capitalize()} - ")
    dist_plot(mc, warmup=warmup, path=BASE_PATH/f"dist_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=False, title_prefix=f"{METHOD.capitalize()} - ")

    PRIOR = "multinorm"
    m = GibbsProbit(prior="multi-norm", intercept=True)
    mc = m.fit(X, Y, n_iter=n_iter, warmup=warmup, seed=SEED)
    trace_plot(mc, path=BASE_PATH/f"trace_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=False, title_prefix=f"{METHOD.capitalize()} - ")
    dist_plot(mc, warmup=warmup, path=BASE_PATH/f"dist_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=False, title_prefix=f"{METHOD.capitalize()} - ")

if __name__ == "__main__":
    # Same number of iterations as paper
    N_ITER, WARMUP = 800, 0
    simul(N_ITER, WARMUP)

    # Larger number of iterations
    N_ITER, WARMUP = 20000, 0
    simul(N_ITER, WARMUP)
