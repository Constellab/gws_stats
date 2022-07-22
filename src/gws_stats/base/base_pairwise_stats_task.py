# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod

import numpy as np
import pandas
from gws_core import (ConfigParams, HeatmapView, ListParam, StrParam, Table,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view, InputSpec, OutputSpec, BadRequestException)
from pandas import concat

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult


@ task_decorator("BasePairwiseStatsTask", hide=True)
class BasePairwiseStatsTask(Task):
    """
    BasePairwiseStatsTask
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 99
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(BasePairwiseStatsResult, human_name="Result", short_description="The output result")}
    config_specs = {
        "column_names":
        ListParam(
            default_value=None, optional=True, human_name="Columns names",
            short_description=f"The columns used for pairwise comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used"),
        "reference_column":
        StrParam(
            default_value=None, optional=True, human_name="Reference column",
            short_description=f"The columns used as reference for pairwise comparison. Only this column is compared with the others.")
    }

    _remove_nan_before_compute = True

    @abstractmethod
    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        return None

    def remove_nan(data):
        pass

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        target_cols = params.get_value("column_names")
        if target_cols:
            data = data.loc[:, target_cols]
        else:
            self.log_info_message(f"No column names given. The first {self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
            data = data.iloc[:, 0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]

        reference_column = params.get_value("reference_column")
        if reference_column:
            if reference_column in data.columns:
                k = data.columns.get_loc(reference_column)
                ref_col_indexes = [ k ]
            else:
                raise BadRequestException(f"The reference column {reference_column} name is not found")
        else:
            ref_col_indexes = range(0, data.shape[1])

        is_nan_log_shown = False
        all_result = None

        for i in ref_col_indexes:
            ref_col = data.columns[i]
            ref_data = data.iloc[:, [i]]

            for j in range(0, data.shape[1]):
                if not reference_column:
                    if j <= i:
                        continue

                target_col = data.columns[j]
                if target_col == ref_col:
                    continue

                target_data = data.iloc[:, [j]]
                current_data = concat(
                    [ref_data, target_data],
                    axis=1
                )
                current_data = current_data.to_numpy().T

                array_sum = np.sum(current_data)
                array_has_nan = np.isnan(array_sum)
                if array_has_nan:
                    if self._remove_nan_before_compute:
                        current_data = [[x for x in y if not np.isnan(x)] for y in current_data]
                    if not is_nan_log_shown:
                        self.log_warning_message("Data contain NaN values. NaN values are omitted.")
                        is_nan_log_shown = True

                stat_result = self.compute_stats(current_data, ref_col, target_col, params)

                if all_result is None:
                    all_result = [stat_result]
                else:
                    all_result = np.vstack((all_result, stat_result))

        t = self.output_specs["result"].get_default_resource_type()
        result = t(result=all_result, input_table=table)
        return {'result': result}
