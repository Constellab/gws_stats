# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Type, List
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_scalar

import arviz as az
import pymc as pm
from gws_core import BadRequestException
from .mc_sampler import MCSampler


class MCLinRegData:
    """ Linear regression data """
    x_data: DataFrame = None
    y_data: DataFrame = None

    def __init__(self, x_data, y_data):
        if not isinstance(x_data, DataFrame):
            raise BadRequestException("Data x_data must be a DataFrame")
        if not isinstance(y_data, DataFrame):
            raise BadRequestException("Data y_data must be a DataFrame")
        self.x_data = x_data
        self.y_data = y_data


class MCLinRegResult:
    """ Linear regression data """
    _traces = None
    _data: MCLinRegData = None

    def __init__(self, traces, data: MCLinRegData):
        self._traces = traces
        self._data = data

    def get_traces(self):
        return self._traces

    def get_slope_traces(self, num_samples=None):
        dfs = []
        for i, colname in enumerate(self._data.x_data.columns):
            data = az.extract(self._traces, num_samples=num_samples, group='posterior',
                              combined=True, var_names=f"slope_{i}")
            dfs.append(DataFrame(data.T, columns=[colname]))
        return pd.concat(dfs, axis=1)

    # def get_intercept_traces(self, num_samples=None):
    #     data = az.extract(self._traces, num_samples=num_samples, group='posterior',
    #                       combined=True, var_names="intercept_0")
    #     df = DataFrame(data.T, columns=self._data.y_data.columns)
    #     return df

    def get_predictions(self, num_samples=None):
        slope = self.get_slope_traces(num_samples=num_samples)
        intercept = None
        # if "intercept_0" in self._traces.posterior:
        #     intercept = self.get_intercept_traces(num_samples=num_samples)

        preds = []
        for i in range(0, num_samples):
            if intercept is None:
                preds.append(self._data.x_data @ slope.iloc[i, :].T)
            else:
                preds.append(intercept + self._data.x_data @ slope.iloc[i, :].T)

        preds = pd.concat(preds, axis=1)
        return preds

    def get_prediction_stats(self, num_samples=None):
        preds = self.get_predictions(num_samples=num_samples)
        pred_std = preds.std(axis=1)
        pred_mean = preds.mean(axis=1)
        pred_stats = pd.concat([pred_mean, pred_std], axis=1)
        pred_stats.columns = ["mean", "std"]
        return pred_stats


class MCLinRegSampler(MCSampler):
    """
    Robust Monte-Carlo based Linear Regression
    See also https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust.html
    """

    _tune = 1000
    _draws = 2000
    _chains = 2
    _model = None

    _slope_priors = None
    _intercept_priors = None
    _sigma_priors = None

    _intercept_names: List[str] = None
    _likelihood_distrib = pm.Normal

    def set_slope_priors(self, priors: list):
        if not isinstance(priors, list):
            raise BadRequestException("Slope priors: a list of dict is expected")
        for i, prior in enumerate(priors):
            prior["name"] = f"slope_{i}"
        self._slope_priors = priors

    # def set_intercept_priors(self, priors: list):
    #     for i, prior in enumerate(priors):
    #         prior["name"] = f"intercept_{i}"
    #     self._intercept_priors = priors

    def set_sigma_prior(self, prior: dict):
        if not isinstance(prior, dict):
            raise BadRequestException("Sigma prior: a dict is expected")
        priors = [prior]
        for i, prior in enumerate(priors):
            prior["name"] = f"sigma_{i}"
        self._sigma_priors = priors

    def set_likelihood_type(self, likelihood_type: Type[pm.Distribution]):
        self._likelihood_distrib = likelihood_type

    def create_model(self, args=None):

        with pm.Model() as model:
            data = self.get_observed_data()
            x_out = data.x_data
            y_out = data.y_data

            slopes = self._build_distribs(self._slope_priors)
            sigmas = self._build_distribs(self._sigma_priors)

            mu_hat = 0
            for i, slope in enumerate(slopes):
                name = x_out.columns[i]
                xi = pm.ConstantData(name, x_out.iloc[:, i])
                mu_hat = mu_hat + xi * slope

            if self._intercept_priors is not None:
                intercepts = self._build_distribs(self._intercept_priors)
                mu_hat = mu_hat + intercepts[0]

            mu_hat = pm.Deterministic("mu", mu_hat)
            self._likelihood_distrib("y", mu=mu_hat, sigma=sigmas[0], observed=y_out)

            return model

    def sample(self, model, random_seed=None):
        with model:
            traces = pm.sample(random_seed=random_seed, draws=self._draws, tune=self._tune, chains=self._chains)
            return MCLinRegResult(traces, data=self.get_observed_data())
