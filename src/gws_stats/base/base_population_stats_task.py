# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod

import numpy as np
import pandas
from gws_core import (BadRequestException, BoolParam, ConfigParams, FloatParam,
                      InputSpec, ListParam, OutputSpec, ParamSet, StrParam,
                      Table, TableUnfolderHelper, Task, TaskInputs,
                      TaskOutputs, task_decorator)
from pandas import concat

from ..base.base_population_stats_result import BasePopulationStatsResult


@ task_decorator("BasePopulationStatsTask", hide=True)
class BasePopulationStatsTask(Task):
    """
    BasePopulationStatsTask

    Performs comparison of multiple columns of a table
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 500

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(BasePopulationStatsResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        "preselected_column_names":
        ParamSet({
            "name": StrParam(
                default_value="", human_name="Pre-selected columns names", optional=True,
                short_description="The name of the column(s) to pre-select"),
            "is_regex": BoolParam(
                default_value=False, human_name="Is text pattern?",
                short_description="Set True if it is a text pattern (regular expression), False otherwise")
        }, human_name="Pre-selected column names", short_description=f"The names of column to pre-select for comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used", optional=True),
        "row_tag_key":
        StrParam(
            default_value=None, optional=True, human_name="Row tag key (for group-wise comparisons)",
            visibility=StrParam.PROTECTED_VISIBILITY,
            short_description="The key of the row tag (representing the group axis) along which one would like to compare each column. This parameter is not used if a `reference column` is given.")
    }

    _remove_nan_before_compute = True

    @abstractmethod
    def compute_stats(self, data, params: ConfigParams):
        """ Compute stats """
        return None

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        selected_cols = params.get_value("preselected_column_names")
        if selected_cols:
            table = table.select_by_column_names(selected_cols)

        row_tag_key = params.get_value("row_tag_key")
        if row_tag_key:
            stat_result = self._row_group_compare(table, params)
        else:
            stat_result = self._column_compare(table, params)

        t = self.output_specs["result"].get_default_resource_type()
        result = t(result=stat_result, input_table=table)
        return {'result': result}

    def _column_compare(self, table, params):
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')
        array_has_nan = data.isnull().sum().sum()
        data = data.to_numpy().T
        if array_has_nan:
            self.log_warning_message("Data contain NaN values. NaN values are omitted.")
            data = [[x for x in y if not np.isnan(x)] for y in data]  # remove nan values

        stat_result = self.compute_stats(data, params)
        stat_result = ["*", stat_result.statistic, stat_result.pvalue]
        stat_result = pandas.DataFrame([stat_result])
        return stat_result

    def _row_group_compare(self, table, params):
        key = params.get_value("row_tag_key")
        data = table.get_data()

        all_stat_result = None
        for k in range(0, data.shape[1]):
            # select each column separately to compare them
            sub_table = table.select_by_column_positions([k])
            # unfold the current column
            sub_table = TableUnfolderHelper.unfold_rows_by_tags(sub_table, [key], 'column_name')

            sub_data = sub_table.get_data()
            sub_data = sub_data.apply(pandas.to_numeric, errors='coerce')
            sub_data = sub_data.to_numpy().T
            array_sum = np.sum(sub_data)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                self.log_warning_message("Data contain NaN values. NaN values are omitted.")
                sub_data = [[x for x in y if not np.isnan(x)] for y in data]  # remove nan values

            # compare all the unfolded columns
            stat_result = self.compute_stats(sub_data, params)
            if all_stat_result is None:
                stat_result = [data.columns[k], stat_result.statistic, stat_result.pvalue]
                all_stat_result = pandas.DataFrame([stat_result])
            else:
                stat_result = [data.columns[k], stat_result.statistic, stat_result.pvalue]
                stat_result = pandas.DataFrame([stat_result])
                all_stat_result = pandas.concat([all_stat_result, stat_result], axis=0, ignore_index=True)

        return all_stat_result
