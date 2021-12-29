# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, HeatmapView, ListParam, StrParam, Table,
                      TableView, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from scipy.stats import f_oneway

from ..base.base_resource import BaseResource
from ..view.stats_boxplot_view import StatsBoxPlotView


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

        print(cdata)
        return Table(cdata)

    @view(view_type=TableView, default_view=True, human_name="StatisticsTable",
          short_description="Table of statistic and p-value")
    def view_statistics_table(self, params: ConfigParams) -> TableView:
        """
        View statistics table
        """

        stats_data = self.get_result()
        table = Table(stats_data)
        return TableView(table)

    @view(view_type=TableView, default_view=True, human_name="ContingencyTable",
          short_description="The contingency table of P-Values",
          specs={
              "metric": StrParam(
                  human_name="metric",
                  allowed_values=["p-value", "statistic"],
                  default_value="p-value")})
    def view_contingency_table(self, params: ConfigParams) -> TableView:
        """
        View contingency table
        """

        table = self._compute_contingency_table(params)
        return TableView(table)

    @view(view_type=HeatmapView, default_view=True, human_name="ContingencyMap",
          short_description="The contingency table of P-Values as HeatMap")
    def view_contingency_map(self, params: ConfigParams) -> TableView:
        """
        View contingency map
        """

        table = self._compute_contingency_table(params)
        return TableView(table)

    @ view(view_type=StatsBoxPlotView, human_name="StatBoxplot",
           short_description="Boxplot of data with statistics and p-value")
    def view_stats_result_as_boxplot(self, params: ConfigParams) -> StatsBoxPlotView:
        """
        View boxplots
        """

        stats_data = self.get_result()
        return StatsBoxPlotView(self.table, stats=stats_data)
