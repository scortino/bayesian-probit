import numpy as np
from scipy.stats import norm


class BaseBayesianProbit:
    def __init__(self, prior=None, intercept=True, epsilon=1e-20):
        if prior is None:
            prior = lambda x: 1.0 # non-informative prior
        self.prior = prior
        self.beta = None
        self.intercept = intercept
        self.epsilon = epsilon

    def __repr__(self):
        attr = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({attr})"

    def get_params(self):
        print(self.__dict__)

    def set_params(self, **params):
        for param_name in params:
            if param_name not in self.__dict__:
                raise ValueError(f"{param_name} is not a valid parameter for {self.__class__.__name__}")
        self.__dict__.update(params)

    def predict(self, X, threshold=0.5):
        if self.beta is None:
            raise Exception("model must be fitted on data before being used for prediction")
        if self.intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        eta = X @ self.beta
        p_hat = norm.cdf(eta)
        preds = np.where(p_hat > threshold, 1, 0)
        return preds
