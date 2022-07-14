# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, ListParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view, InputSpec, OutputSpec)
from scipy.stats import f_oneway

from ..base.base_stats_result import BaseStatsResult

# *****************************************************************************
#
# PearsonCorrelationResult
#
# *****************************************************************************


@resource_decorator("PearsonCorrelationResult", human_name="Pearson correlation result",
                    short_description="Result of the person correlation analysis", hide=True)
class PearsonCorrelationResult(BaseStatsResult):
    pass

# *****************************************************************************
#
# PearsonCorrelation
#
# *****************************************************************************


@task_decorator("PearsonCorrelation", human_name="Pearson Correlation",
                short_description="Compute the pearson correlation coefficient with p-value")
class PearsonCorrelation(Task):
    """
    ...

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: the one-way ANOVA F statistic, and the corresponding p-value.

    * Config Parameters:


    For more details, see ...
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 99

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(OneWayAnovaResult, human_name="Result", short_description="The output result")}
    config_specs = {
        "column_names": ListParam(
            default_value=[], human_name="Column (group) names",
            short_description="The names of the columns that represent the groups to compare")
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        column_names = params.get_value("column_names", [])
        if not column_names:
            self.log_info_message("No column names given. The first 3 columns are used.")
            column_names = data.columns[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]

        data = data[column_names].to_numpy().T
        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)

        if array_has_nan:
            self.log_warning_message("Data contain NaN values. NaN values are omitted.")
            # removing NaN values
            data = [[x for x in y if not np.isnan(x)] for y in data]

        stat_result = f_oneway(*data)
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        result = OneWayAnovaResult(result=stat_result, input_table=table)
        return {'result': result}
