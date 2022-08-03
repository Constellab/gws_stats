# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod

import numpy as np
import pandas
from gws_core import (BadRequestException, ConfigParams, HeatmapView,
                      InputSpec, ListParam, OutputSpec, StrParam, Table, Task,
                      TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view)
from pandas import concat

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult


@ task_decorator("BasePairwiseStatsTask", hide=True)
class BasePairwiseStatsTask(Task):
    """
    BasePairwiseStatsTask
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 99
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(BasePairwiseStatsResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        "reference_column":
        StrParam(
            default_value=None, optional=True, human_name="Reference column",
            short_description="The columns used as reference for pairwise comparison. Only this column is compared with the others."),
        "column_names":
        ListParam(
            default_value=None, optional=True, human_name="Selected columns names",
            short_description=f"The columns selected for pairwise comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used"),
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

        reference_column = params.get_value("reference_column")
        selected_cols = params.get_value("column_names")

        if reference_column:
            if reference_column in data.columns:
                reference_columns = [reference_column]
            else:
                raise BadRequestException(f"The reference column {reference_column} name is not found")
        else:
            reference_columns = list(set(data.columns[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]))

        if selected_cols:
            selected_cols = list(set([*selected_cols, *reference_columns]))
        else:
            self.log_info_message(
                f"No column names given. The first {self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
            selected_cols = data.columns[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]
            selected_cols = list(set([*selected_cols, *reference_columns]))

        data = data.loc[:, selected_cols]

        is_nan_log_shown = False
        all_result = None

        for ref_col in data.columns:
            if ref_col not in reference_columns:
                continue
            i = data.columns.get_loc(ref_col)
            ref_data = data.iloc[:, [i]]

            for target_col in data.columns:
                j = data.columns.get_loc(target_col)
                if not reference_column:
                    if j <= i:
                        continue
                # if target_col == ref_col:
                #     continue

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
                    all_result = pandas.DataFrame(stat_result).T
                else:
                    df = pandas.DataFrame(stat_result).T
                    # np.vstack((all_result, stat_result))
                    all_result = pandas.concat([all_result, df], axis=0, ignore_index=True)

        t = self.output_specs["result"].get_default_resource_type()
        result = t(result=all_result, input_table=table)
        return {'result': result}
