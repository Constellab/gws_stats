# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoxPlotView, ConfigParams, HeatmapView, ListParam,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from scipy.stats import f_oneway

from ..base.base_resource import BaseResource


@resource_decorator("BasePairwiseStatsResult", hide=True)
class BasePairwiseStatsResult(BaseResource):

    PVALUE_NAME = "P-Value"
    STATISTICS_NAME = "Statistic"
    REFERENCE_NAME = "Reference"
    COMPARED_NAME = "Compared"

    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        cls = type(self)
        columns = [
            cls.REFERENCE_NAME,
            cls.COMPARED_NAME,
            cls.STATISTICS_NAME,
            cls.PVALUE_NAME
        ]

        data = DataFrame(stat_result, columns=columns)
        return data

    def _compute_contingency_table(self, params: ConfigParams):
        stats_data = self.get_result()
        metric = params.get_value("metric")
        cls = type(self)
        columns = [
            *stats_data.loc[:, cls.REFERENCE_NAME].values.tolist(),
            *stats_data.loc[:, cls.COMPARED_NAME].values.tolist()
        ]
        columns = list(set(columns))
        n = len(columns)
        cdata = DataFrame(np.ones([n, n]), columns=columns, index=columns)
        for i in range(0, stats_data.shape[0]):
            col1 = stats_data.loc[i, cls.REFERENCE_NAME]
            col2 = stats_data.loc[i, cls.COMPARED_NAME]
            if metric == "p-value":
                cdata.at[col1, col2] = stats_data.loc[i, cls.PVALUE_NAME]
                cdata.at[col2, col1] = stats_data.loc[i, cls.PVALUE_NAME]
            else:
                cdata.at[col1, col2] = stats_data.loc[i, cls.STATISTICS_NAME]
                cdata.at[col2, col1] = stats_data.loc[i, cls.STATISTICS_NAME]

        for i in range(0, cdata.shape[0]):
            for j in range(i, cdata.shape[0]):
                cdata.iloc[j, i] = np.nan

        return cdata

    @view(view_type=TabularView, default_view=True, human_name="Statistics table",
          short_description="Table of statistic and p-value", specs={})
    def view_statistics_table(self, params: ConfigParams) -> TabularView:
        """
        View statistics table
        """

        data = self.get_result()
        t_view = TabularView()
        t_view.set_data(data=data)
        return t_view

    @view(view_type=TabularView, default_view=True, human_name="Contingency table",
          short_description="The contingency table of P-Values",
          specs={
              "metric": StrParam(
                  human_name="metric",
                  allowed_values=["p-value", "statistic"],
                  default_value="p-value")})
    def view_contingency_table(self, params: ConfigParams) -> TabularView:
        """
        View contingency table
        """

        data = self._compute_contingency_table(params)
        t_view = TabularView()
        t_view.set_data(data=data)
        return t_view

    @view(view_type=HeatmapView, default_view=True, human_name="Contingency map",
          short_description="The contingency table of P-Values as HeatMap", specs={})
    def view_contingency_map(self, params: ConfigParams) -> HeatmapView:
        """
        View contingency map
        """

        data = self._compute_contingency_table(params)
        t_view = HeatmapView()
        t_view.set_data(data=data)
        return t_view
