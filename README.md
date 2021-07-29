# Bayesian Estimation of a Probit Regression Model

This repository contains the code related to our final project for class [20592 Statistics and Probability](http://didattica.unibocconi.it/ts/tsn_anteprima.php?cod_ins=20592&anno=2021&IdPag=6352) at Bocconi University. The project aims at illustrating the use of Bayesian methods for estimating the coefficient of a probit regression model for binary outcomes. In particular, the code in this repository implements the Metropolis algorithm and the auxiliary variable Gibbs sampler and runs simulations to show how they can be used to estimate the coefficients of a probit regression model.

## Content

The current repository is structured as follows:
- `data/finney47.csv`: dataset used for testing the methods implemented in `metropolis.py` and `gibbs.py` (TODO: add citation);
- `images/`: contain trace plots and distribution plots obtained from the simulations in `gibbs_simulation.py` and `metropolis_simulation.py`;
- `project/`:
    - `base.py`: implements BaseBayesianProbit class;
    - `gibbs.py`: implements auxiliary variable Gibbs sampler in `GibbsProbit` class;
    - `metropolis.py`: implements Metropolis algorithm in `MetropolisProbit` class;
    - `utils.py`: implements utility functions for loading the data and obtaining the plots shown in `images`;
- `bayesian_probit.pdf`: final PDF report;
- `gibbs_simulation.py`: main simulation script for testing the auxiliary variable Gibbs sampler;
- `metropolis_simulation.py`: main simulation script for testing the Metropolis algorithm;
- `README.md`;
- `requirements.txt`.

## Requirements

The Python packages required to run the simulations can be installed from withing the repository by using the `pip` package manager:

```console
pip install -r requirements.txt
```
## Simulations

The simulations performed for the project can be replicated by running:

```
python gibbs_simulation.py
python metropolis_simulation.py
```

If not already present in `images/`, new trace plots and distribution plots for the coefficients of the probit model sampled by the two methods mentioned above are generated and stored.

## Authors

- [Stefano Cortinovis](https://github.com/scortino)
- [Daniele Micheletti](https://github.com/danielemiche)
- Andrea Teruzzi
- Leonardo Yang