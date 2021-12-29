# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoolParam, ConfigParams, StrParam, Table,
                      resource_decorator, task_decorator)
from scipy.stats import ttest_rel

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# TTestTwoRelatedSamplesResult
#
# *****************************************************************************


@resource_decorator("TTestTwoRelatedSamplesResult", hide=True)
class TTestTwoRelatedSamplesResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "T-Statistic"

# *****************************************************************************
#
# TTestTwoRelatedSamples
#
# *****************************************************************************


@task_decorator("TTestTwoRelatedSamples")
class TTestTwoRelatedSamples(BasePairwiseStatsTask):
    """
    Compute the T-test for the means of related samples, from a given reference sample.

    This test is a two-sided (or one-side) test for the null hypothesis that 2 independent samples have identical average (expected) values.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the T-statistic, and the p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    input_specs = {'table': Table}
    output_specs = {'result': TTestTwoRelatedSamplesResult}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=["two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing.")
    }
    _remove_nan_before_compute = False

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        alternative = params.get_value("alternative_hypothesis")
        stat_result = ttest_rel(*current_data, nan_policy='omit', alternative=alternative)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        return stat_result
