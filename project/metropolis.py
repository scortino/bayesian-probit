import numpy as np
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from .base import BaseBayesianProbit


class MetropolisProbit(BaseBayesianProbit):
    def __init__(self, prior=None, intercept=True, epsilon=1e-20):
        if prior is None:
            prior = lambda x: 1.0 # non-informative prior
        self.prior = prior
        super(MetropolisProbit, self).__init__(intercept, epsilon)

    def fit(self, X, Y, beta_0=None, n_iter=2000, warmup=200, return_chain=True, seed=None):
        np.random.seed(seed)
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y if beta_0 is None else beta_0
        betas = [beta]
        eta = X @ beta
        p_hat = norm.cdf(eta)
        L = np.prod(((p_hat) ** Y) * ((1 - p_hat) ** (1 - Y)))
        for _ in tqdm(range(n_iter)):
            I = X.T @ np.diag((norm.pdf(eta) ** 2) / (p_hat * (1 - p_hat) + self.epsilon)) @ X
            cov = np.linalg.inv(I)
            beta_star = multivariate_normal.rvs(beta, cov)
            eta_star = X @ beta_star
            p_hat_star = norm.cdf(eta_star)
            L_star = np.prod(((p_hat_star) ** Y) * ((1 - p_hat_star) ** (1 - Y)))
            alpha = (self.prior(beta_star) * L_star) / (self.prior(beta) * L)
            if np.random.rand() < alpha:
                beta = beta_star
                eta = eta_star
                p_hat = p_hat_star
                L = L_star
            betas.append(beta)
        self.beta = sum(betas[warmup:]) / (n_iter - warmup) # bayesian estimator
        if return_chain:
            return betas
