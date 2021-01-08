import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal

from project.metropolis import MetropolisProbit
from project.utils import load_finney47, trace_plot, dist_plot, ar_plot


METHOD = 'metropolis'
BASE_PATH = Path('./images')
SEED = 42
REPLACE = True

# Finney (1947), see paper p. 675
X, Y = load_finney47()

# Function that, given n_iter and warmup, runs the algorithm on the data for three different priors
def simul(n_iter, warmup):
    PRIOR = "noinfo"
    m = MetropolisProbit(prior=None, intercept=True) # non-informative improper prior
    mc, accepted = m.fit(X, Y, n_iter=n_iter, warmup=warmup, seed=SEED)
    trace_plot(mc, path=BASE_PATH/f"trace_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")
    dist_plot(mc, warmup=warmup, path=BASE_PATH/f"dist_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")
    ar_plot(accepted, warmup=warmup, path=BASE_PATH/f"ar_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")

    PRIOR = "highvar"
    m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([10.0, 10.0, 10.0]))) # independent normal prior with high variance
    mc, accepted = m.fit(X, Y, n_iter=n_iter, warmup=warmup, seed=SEED)
    trace_plot(mc, path=BASE_PATH/f"trace_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")
    dist_plot(mc, warmup=warmup, path=BASE_PATH/f"dist_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")

    PRIOR = "lowvar"
    m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([1.0, 1.0, 1.0]))) # independent normal prior with low variance
    mc, accepted = m.fit(X, Y, n_iter=n_iter, warmup=warmup, seed=SEED)
    trace_plot(mc, path=BASE_PATH/f"trace_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")
    dist_plot(mc, warmup=warmup, path=BASE_PATH/f"dist_{METHOD}_{PRIOR}_{n_iter}_warmup_{warmup}.png", replace=REPLACE, title_prefix=f"{METHOD.capitalize()} - ")

if __name__ == "__main__":
    # Same number of iterations as paper
    N_ITER, WARMUP = 200, 0
    simul(N_ITER, WARMUP)

    # Same number of iterations as paper
    N_ITER, WARMUP = 800, 0
    simul(N_ITER, WARMUP)

    # Larger number of iterations
    N_ITER, WARMUP = 20000, 0
    simul(N_ITER, WARMUP)
