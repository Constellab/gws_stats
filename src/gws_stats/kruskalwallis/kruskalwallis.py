# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, ListParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from scipy.stats import kruskal

from ..base.base_stats_result import BaseStatsResult

# *****************************************************************************
#
# KruskalWallisResult
#
# *****************************************************************************


@resource_decorator("KruskalWallisResult", hide=True)
class KruskalWallisResult(BaseStatsResult):
    pass

# *****************************************************************************
#
# KruskalWallis
#
# *****************************************************************************


@task_decorator("KruskalWallis")
class KruskalWallis(Task):
    """
    Compute the Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the population
    median of all of the groups are equal.  It is a non-parametric version of
    ANOVA.  The test works on 2 or more independent samples, which may have
    different sizes.  Note that rejecting the null hypothesis does not
    indicate which of the groups differs.  Post hoc comparisons between
    groups are required to determine which groups are different.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: the Kruskal-Wallis H statistic, corrected for ties, and the p-value for the test using the assumption that H has a chi
       square distribution. The p-value returned is the survival function of the chi square distribution evaluated at H.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    Note: due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small.  A typical rule is
    that each sample must have at least 5 measurements.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    """
    input_specs = {'table': Table}
    output_specs = {'result': KruskalWallisResult}
    config_specs = {
        "column_names": ListParam(
            default_value=[], human_name="Column names",
            short_description="The names of the columns that represent the groups to compare")
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        column_names = params.get_value("column_names", [])
        if not column_names:
            column_names = data.columns[0:3]

        data = data[column_names].to_numpy().T
        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)

        if array_has_nan:
            self.log_warning_message("Data contain NaN values. NaN values are omitted.")
        stat_result = kruskal(*data, nan_policy='omit')

        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        result = KruskalWallisResult(result=stat_result, input_table=table)
        return {'result': result}
