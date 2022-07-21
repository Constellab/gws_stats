# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoxPlotView, ConfigParams, HeatmapView, ListParam,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from pandas import DataFrame

from ..base.base_resource import BaseResource


@resource_decorator("BasePairwiseStatsResult", hide=True)
class BasePairwiseStatsResult(BaseResource):

    PVALUE_NAME = "P-Value"
    STATISTICS_NAME = "Statistic"
    REFERENCE_NAME = "Reference"
    COMPARED_NAME = "Compared"

    STATISTIC_TABLE_NAME = "Statistics table"
    PVALUE_CONTINGENCY_TABLE_NAME = "Contingency table - PValue"
    STATISTICS_CONTINGENCY_TABLE_NAME = "Contingency table - Statistics"

    def __init__(self, result=None, input_table: Table = None):
        super().__init__(result=result, input_table=input_table)
        if result is not None:
            self._create_statistics_table()
            self._create_contingency_table("p-value")
            self._create_contingency_table("statictic")

    def get_statistics_table(self) -> DataFrame:
        if self.resource_exists(self.STATISTIC_TABLE_NAME):
            return self.get_resource(self.STATISTIC_TABLE_NAME)
        else:
            return None

    def _create_statistics_table(self) -> DataFrame:
        stat_result = self.get_result()
        columns = [
            self.REFERENCE_NAME,
            self.COMPARED_NAME,
            self.STATISTICS_NAME,
            self.PVALUE_NAME
        ]
        data = DataFrame(stat_result, columns=columns)
        table = Table(data=data)
        table.name = self.STATISTIC_TABLE_NAME
        self.add_resource(table)

    def _create_contingency_table(self, metric):
        stats_data = self.get_statistics_table().get_data()
        #metric = params.get_value("metric")
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

        table = Table(cdata)
        if metric == "p-value":
            table.name = self.PVALUE_CONTINGENCY_TABLE_NAME
        else:
            table.name = self.STATISTICS_CONTINGENCY_TABLE_NAME
        self.add_resource(table)

    def get_contingency_table(self, metric):
        if metric == "p-value":
            name = self.PVALUE_CONTINGENCY_TABLE_NAME
        else:
            name = self.STATISTICS_CONTINGENCY_TABLE_NAME
        return self.get_resouce(name)