import numpy as np
from scipy.stats import multivariate_normal, norm

from .base import BaseBayesianProbit


class MetropolisProbit(BaseBayesianProbit):
    def fit(self, X, Y, beta_0=None, return_chain=True, n_iter=2000, warmup=200):
        beta = np.zeros(X.shape[1] + self.intercept) if beta_0 is None else beta_0
        betas = [beta]
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = X @ beta
        p_hat = norm.cdf(eta)
        ell = np.prod(((p_hat) ** Y) * ((1 - p_hat) ** (1 - Y)))
        for _ in range(n_iter):
            I = X.T @ np.diag((norm.pdf(eta) ** 2) / (p_hat * (1 - p_hat) + self.epsilon)) @ X
            cov = np.linalg.inv(I)
            beta_star = multivariate_normal.rvs(beta, cov)
            eta_star = X @ beta_star
            p_hat_star = norm.cdf(eta_star)
            ell_star = np.prod(((p_hat_star) ** Y) * ((1 - p_hat_star) ** (1 - Y)))
            alpha = (self.prior(beta_star) * ell_star) / (self.prior(beta) * ell) # non-informative prior
            if np.random.rand() < alpha:
                beta = beta_star
                eta = eta_star
                p_hat = p_hat_star
                ell = ell_star
            betas.append(beta)
        self.beta = sum(betas[warmup:]) / (n_iter - warmup) # bayesian estimator
        if return_chain:
            return betas
