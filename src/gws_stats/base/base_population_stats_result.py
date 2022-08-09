# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import Table, resource_decorator
from pandas import DataFrame

from ..base.base_resource import BaseResource


@resource_decorator("BasePopulationStatsResult", hide=True)
class BasePopulationStatsResult(BaseResource):
    """ BasePopulationStatsResult """

    PVALUE_NAME = "PValue"
    STATISTICS_NAME = "Statistic"
    STATISTIC_TABLE_NAME = "Statistics table"

    def __init__(self, result=None, input_table: Table = None):
        super().__init__(result=result, input_table=input_table)
        if result is not None:
            self._create_statistics_table()

    def get_statistics_table(self) -> DataFrame:
        if self.resource_exists(self.STATISTIC_TABLE_NAME):
            return self.get_resource(self.STATISTIC_TABLE_NAME)
        else:
            return None

    def _create_statistics_table(self) -> DataFrame:
        stat_result = self.get_result()
        columns = [
            "Columns",
            self.STATISTICS_NAME,
            self.PVALUE_NAME,
        ]

        table = Table(data=stat_result, column_names=columns)
        table.name = self.STATISTIC_TABLE_NAME
        self.add_resource(table)
