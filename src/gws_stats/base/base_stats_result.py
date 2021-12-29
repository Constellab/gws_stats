# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (ConfigParams, ListParam, Table, TableView,
                      resource_decorator, view)
from pandas import DataFrame

from .base_resource import BaseResource


@resource_decorator("BaseStatsResult", hide=True)
class BaseStatsResult(BaseResource):

    PVALUE_NAME = "P-Value"
    FSTATISTICS_NAME = "F-Statistic"

    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = [
            BaseStatsResult.FSTATISTICS_NAME,
            BaseStatsResult.PVALUE_NAME
        ]
        data = DataFrame([stat_result], columns=columns)
        return data

    @view(view_type=TableView, default_view=True, human_name="StatisticsTable",
          short_description="Table of statistic and p-value")
    def view_statistics_table(self, params: ConfigParams) -> dict:
        """
        View stats Table
        """

        stat_result = self.get_result()
        table = Table(stat_result)
        return TableView(table)
