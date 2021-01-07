import numpy as np
from scipy.stats import multivariate_normal, truncnorm
from tqdm import tqdm

from .base import BaseBayesianProbit


class GibbsProbit(BaseBayesianProbit):
    def __init__(self, prior="non-informative", intercept=True, epsilon=1e-20):
        if prior not in ["non-informative", "multi-norm"]:
            raise ValueError("Only improper non-informative and standard multivariate normal priors are supported")
        self.prior = prior
        super(GibbsProbit, self).__init__(intercept, epsilon)

    def fit(self, X, Y, beta_0=None, return_chain=True, n_iter=2000, warmup=200, seed=None):
        np.random.seed(seed)
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y if beta_0 is None else beta_0
        betas = [beta]
        Z = np.zeros_like(Y)
        m = Y == 1
        if self.prior == "non-informative":
            XprimeX_inv = np.linalg.inv(X.T @ X)
        elif self.prior == "multi-norm":
            # TODO: allow user to specify beta_star and b_star (in __init__?)
            beta_star = np.zeros_like(beta)
            b_star_inv = np.linalg.inv(np.eye(len(beta)))
            beta_tilde_base = np.linalg.inv(b_star_inv + X.T @ X)
            b_tilde = np.linalg.inv(b_star_inv + X.T @ X)
        for _ in tqdm(range(n_iter)):
            eta = X @ beta
            Z[m] = truncnorm.rvs(-eta[m], np.inf, loc=eta[m], scale=1)
            Z[~m] = truncnorm.rvs(-np.inf, -eta[~m], loc=eta[~m], scale=1)
            if self.prior == "non-informative":
                beta = multivariate_normal.rvs(XprimeX_inv @ X.T @ Z, XprimeX_inv)
            elif self.prior == "multi-norm":
                beta_tilde = beta_tilde_base @ (b_star_inv @ beta_star + X.T @ Z)
                beta = multivariate_normal.rvs(beta_tilde, b_tilde)
            betas.append(beta)
        self.beta = sum(betas[warmup:]) / (n_iter - warmup) # bayesian estimator
        if return_chain:
            return betas
