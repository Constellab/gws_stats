# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BadRequestException, BoxPlotView, ConfigParams,
                      HeatmapView, ListParam, StrParam, Table, TabularView,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view)
from pandas import DataFrame

from ..base.base_resource import BaseResource


@resource_decorator("BasePairwiseStatsResult", hide=True)
class BasePairwiseStatsResult(BaseResource):

    PVALUE_NAME = "PValue"
    ADJUSTED_PVALUE_NAME = "Adjusted_PValue"
    STATISTICS_NAME = "Statistic"
    REFERENCE_NAME = "Reference"
    COMPARED_NAME = "Compared"

    STATISTIC_TABLE_NAME = "Statistics table"
    PVALUE_CONTINGENCY_TABLE_NAME = "Contingency table - PValue"
    ADJUSTED_PVALUE_CONTINGENCY_TABLE_NAME = "Contingency table - Adjusted PValue"
    STATISTICS_CONTINGENCY_TABLE_NAME = "Contingency table - Statistics"

    def __init__(self, result=None, input_table: Table = None):
        super().__init__(result=result, input_table=input_table)
        if result is not None:
            self._create_statistics_table()
            self._create_contingency_table(self.PVALUE_NAME)
            self._create_contingency_table(self.ADJUSTED_PVALUE_NAME)
            self._create_contingency_table(self.STATISTICS_NAME)

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
            self.PVALUE_NAME,
            self.ADJUSTED_PVALUE_NAME
        ]

        table = Table(data=stat_result, column_names=columns)
        table.name = self.STATISTIC_TABLE_NAME
        self.add_resource(table)

    def _create_contingency_table(self, metric):
        stats_data = self.get_statistics_table().get_data()
        columns = [
            *stats_data.loc[:, self.REFERENCE_NAME].values.tolist(),
            *stats_data.loc[:, self.COMPARED_NAME].values.tolist()
        ]
        columns = sorted(list(set(columns)))
        n = len(columns)

        cdata = DataFrame(np.ones([n, n]), columns=columns, index=columns)
        for i in range(0, stats_data.shape[0]):
            col1 = stats_data.loc[i, self.REFERENCE_NAME]
            col2 = stats_data.loc[i, self.COMPARED_NAME]
            if metric.lower() == self.PVALUE_NAME.lower():
                cdata.at[col1, col2] = stats_data.loc[i, self.PVALUE_NAME]
                cdata.at[col2, col1] = stats_data.loc[i, self.PVALUE_NAME]
            elif metric.lower() == self.ADJUSTED_PVALUE_NAME.lower():
                cdata.at[col1, col2] = stats_data.loc[i, self.ADJUSTED_PVALUE_NAME]
                cdata.at[col2, col1] = stats_data.loc[i, self.ADJUSTED_PVALUE_NAME]
            elif metric.lower() == self.STATISTICS_NAME.lower():
                cdata.at[col1, col2] = stats_data.loc[i, self.STATISTICS_NAME]
                cdata.at[col2, col1] = stats_data.loc[i, self.STATISTICS_NAME]
            else:
                raise BadRequestException(f"Cannot create contingency table. Invalid metric '{metric}'.")

        for i in range(0, cdata.shape[0]):
            for j in range(i, cdata.shape[0]):
                cdata.iloc[j, i] = np.nan

        # remove
        ref_columns = sorted(list(set(stats_data.loc[:, self.REFERENCE_NAME].values.tolist())))
        comp_columns = sorted(list(set(stats_data.loc[:, self.COMPARED_NAME].values.tolist())))
        cdata = cdata.loc[ref_columns, comp_columns]

        table = Table(cdata)

        if metric.lower() == self.PVALUE_NAME.lower():
            table.name = self.PVALUE_CONTINGENCY_TABLE_NAME
        elif metric.lower() == self.ADJUSTED_PVALUE_NAME.lower():
            table.name = self.ADJUSTED_PVALUE_CONTINGENCY_TABLE_NAME
        elif metric.lower() == self.STATISTICS_NAME.lower():
            table.name = self.STATISTICS_CONTINGENCY_TABLE_NAME
        else:
            raise BadRequestException(f"Cannot create contingency table. Invalid metric '{metric}'.")
        self.add_resource(table)

    def get_contingency_table(self, metric):
        """ Get the contingency table """
        if metric.lower() == self.PVALUE_NAME.lower():
            name = self.PVALUE_CONTINGENCY_TABLE_NAME
        elif metric.lower() == self.ADJUSTED_PVALUE_NAME.lower():
            name = self.ADJUSTED_PVALUE_CONTINGENCY_TABLE_NAME
        elif metric.lower() == self.STATISTICS_NAME.lower():
            name = self.STATISTICS_CONTINGENCY_TABLE_NAME
        else:
            raise BadRequestException(f"Cannot find contingency table. Invalid metric '{metric}'.")
        return self.get_resource(name)
