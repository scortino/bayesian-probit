import numpy as np
from scipy.stats import multivariate_normal, truncnorm

from .base import BaseBayesianProbit


class GibbsProbit(BaseBayesianProbit):
    def __init__(self, prior="non-informative", intercept=True, epsilon=1e-20):
        if prior not in ["non-informative", "multi-norm"]:
            raise ValueError("Only improper non-informative and standard multivariate normal priors are supported")
        self.prior = prior
        super(GibbsProbit, self).__init__(intercept, epsilon)

    def fit(self, X, Y, beta_0=None, return_chain=True, n_iter=2000, warmup=200):
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y if beta_0 is None else beta_0
        betas = [beta]
        for _ in range(n_iter):
            Z = truncnorm.rvs(0, np.inf, X @ beta, 1) * (2 * Y - 1)
            if self.prior == "non-informative":
                beta = multivariate_normal.rvs(np.linalg.inv(X.T @ X) @ X.T @ Z, np.linalg.inv(X.T @ X))
            elif self.prior == "multi-norm":
                # TODO: allow user to specify beta_star and b_star (in __init__?)
                beta_star = np.zeros_like(beta)
                b_star_inv = np.linalg.inv(np.eye(len(beta)))
                beta_tilde = np.linalg.inv(b_star_inv + X.T @ X) @ (b_star_inv @ beta_star + X.T @ Z)
                b_tilde = np.linalg.inv(b_star_inv + X.T @ X)
                beta = multivariate_normal.rvs(beta_tilde, b_tilde)
            betas.append(beta)
        self.beta = sum(betas[warmup:]) / (n_iter - warmup) # bayesian estimator
        if return_chain:
            return betas
