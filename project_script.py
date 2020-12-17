import numpy as np
from scipy.stats import multivariate_normal

from project.metropolis import MetropolisProbit
from project.utils import trace_plot


# Finney (1947), see paper p. 675
dataset = np.loadtxt("./data/finney47.csv", delimiter=",", skiprows=1)
X = dataset[:, 1:]
Y = dataset[:, 0]

m = MetropolisProbit(prior=None, intercept=True) # non-informative improper prior
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_noinfo_2000.png")

m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([10.0, 10.0, 10.0]))) # independent normal prior with high variance
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_highvar_2000.png")

m = MetropolisProbit(prior=lambda beta: multivariate_normal.pdf(beta, mean=[0.0, 0.0, 0.0], cov=np.diag([1.0, 1.0, 1.0]))) # independent normal prior with low variance
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_lowvar_2000.png")
