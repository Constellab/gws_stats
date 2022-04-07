# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, ListParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from scipy.stats import f_oneway

from ..base.base_stats_result import BaseStatsResult

# *****************************************************************************
#
# AnovaResult
#
# *****************************************************************************


@resource_decorator("OneWayAnovaResult", hide=True)
class OneWayAnovaResult(BaseStatsResult):
    pass

# *****************************************************************************
#
# OneWayAnova
#
# *****************************************************************************


@task_decorator("OneWayAnova")
class OneWayAnova(Task):
    """
    Compute the one-way ANOVA test for multiple samples.

    The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
    The test is applied to samples from two or more groups, possibly with differing sizes. It is a parametric
    version of the Kruskal-Wallis test.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: the one-way ANOVA F statistic, and the corresponding p-value.

    * Config Parameters:

    Note: the ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test
    or the Alexander-Govern test although with some loss of power.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """

    input_specs = {'table': Table}
    output_specs = {'result': OneWayAnovaResult}
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
            column_names = data.columns[0:3]

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
