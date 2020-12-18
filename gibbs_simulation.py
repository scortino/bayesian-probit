from project.gibbs import GibbsProbit
from project.utils import load_finney47, trace_plot


# Finney (1947), see paper p. 675
X, Y = load_finney47()

m = GibbsProbit(prior="non-informative", intercept=True)
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_gibbs_noinfo_2000.png", replace=False, title_prefix="Gibbs - ")

m = GibbsProbit(prior="multi-norm", intercept=True)
mc = m.fit(X, Y, return_chain=True, n_iter=2000)
preds = m.predict(X)
trace_plot(mc, path="./images/trace_gibbs_multinorm_2000.png", replace=False, title_prefix="Gibbs - ")
