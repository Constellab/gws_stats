
from gws_core import ConfigParams, Table, TabularView, resource_decorator, view
from pandas import DataFrame

from .base_resource import BaseResource


@resource_decorator("BaseStatsResult", hide=True)
class BaseStatsResult(BaseResource):

    PVALUE_NAME = "PValue"
    STATISTICS_NAME = "Statistic"

    FULL_STATISTIC_TABLE_NAME = "Statistics table"

    def __init__(self, result=None, input_table: Table = None):
        super().__init__(result=result, input_table=input_table)
        if result is not None:
            self._create_statistics_table()

    def get_full_statistics_table(self) -> DataFrame:
        if self.resource_exists(self.FULL_STATISTIC_TABLE_NAME):
            return self.get_resource(self.FULL_STATISTIC_TABLE_NAME)
        else:
            return None

    def _create_statistics_table(self) -> DataFrame:
        stat_result = self.get_result()
        columns = [
            self.STATISTICS_NAME,
            self.PVALUE_NAME
        ]
        table = Table(data=stat_result, column_names=columns)
        table.name = self.FULL_STATISTIC_TABLE_NAME
        self.add_resource(table)

    # @view(view_type=TabularView, default_view=True, human_name="Statistics table",
    #       short_description="Table of statistic and p-value", specs={})
    # def view_statistics_table(self, params: ConfigParams) -> dict:
    #     """
    #     View stats Table
    #     """

    #     stat_result = self.get_result()
    #     t_view = TabularView()
    #     t_view.set_data(data=stat_result)
    #     return t_view
