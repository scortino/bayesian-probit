import numpy as np
from scipy.stats import multivariate_normal

from project.metropolis import MetropolisProbit
from project.utils import load_finney47, trace_plot


# Finney (1947), see paper p. 675
X, Y = load_finney47()

m = MetropolisProbit(prior=None, intercept=True) # non-informative improper prior
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_metropolis_noinfo_2000.png", replace=False, title_prefix="Metropolis - ")

m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([10.0, 10.0, 10.0]))) # independent normal prior with high variance
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_metropolis_highvar_2000.png", replace=False, title_prefix="Metropolis - ")

m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([1.0, 1.0, 1.0]))) # independent normal prior with low variance
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_metropolis_lowvar_2000.png", replace=False, title_prefix="Metropolis - ")
