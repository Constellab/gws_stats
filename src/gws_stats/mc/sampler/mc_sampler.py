# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import BadRequestException
from abc import abstractmethod
from typing import Tuple, Type

import numpy as np
import pymc as pm


class MCSampler:
    """
    General class for Monte Carlo sampling
    """

    _cached_data = None
    _tune = 1000
    _draws = 1000
    _chains = 2

    def __init__(self):
        self._model = pm.Model()

    # -- B --

    @staticmethod
    def _build_distribs(priors: list):
        dist_list = []

        for val in priors:
            name = val.pop("name")
            func = val.pop("func")
            if func == "Normal":
                dist = pm.Normal(name, **val)
            elif func == "TruncatedNormal":
                lb = val.get("lower", -np.inf)
                ub = val.get("upper", np.inf)
                BoundedDist = pm.Bound(pm.TruncatedNormal, lower=lower, upper=upper)
                dist = pm.BoundedDist(name, **val)
            elif func == "HalfNormal":
                dist = pm.HalfNormal(name, **val)
            elif func == "HalfCauchy":
                dist = pm.HalfCauchy(name, **val)
            else:
                raise BadRequestException(f"The distribution function '{func}' is unknown")

            dist_list.append(dist)

        return dist_list

    # -- C --

    @abstractmethod
    def create_model(self, args=None):
        """ Create the model """

    # -- L --

    def load_data(self):
        """
        Load the data

        Override this method to tell the class how the load data
        """

        return None

    # -- G --

    def get_data(self):
        """ Returns the data """
        if self._cached_data is None:
            data = self.load_data()
            if data is None:
                raise BadRequestException("No data defined. Please set data or define the `load_data()` method")

            self._cached_data = data
        return self._cached_data

    def get_observed_data(self):
        """
        Returns the observed data

        To override if required
        """
        return self.get_data()

    # -- S --

    def set_data(self, data):
        """ Set the data """
        self._cached_data = data

    def sample(self, model, random_seed=None):
        vars_list = list(model.values_to_rvs.keys())[:-1]
        # inference
        with model:
            trace = pm.sample(step=[pm.Slice(vars_list)], random_seed=random_seed,
                              tune=self._tune, draws=self._draws, chains=self._chains)
        return trace

    # -- T --

    def trace(self, random_seed=None, args=None):
        model = self.create_model(args=args)
        trace = self.sample(model, random_seed=random_seed)
        return trace
