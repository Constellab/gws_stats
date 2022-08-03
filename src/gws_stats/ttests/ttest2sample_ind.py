# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (BoolParam, ConfigParams, InputSpec, OutputSpec, StrParam,
                      Table, resource_decorator, task_decorator)
from scipy.stats import ttest_ind

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# TTestTwoIndepSamplesResult
#
# *****************************************************************************


@resource_decorator("TTestTwoIndepSamplesResult", human_name="T-test two indep. samples result",
                    short_description="Result of independent samples Student test (T-Test)", hide=True)
class TTestTwoIndepSamplesResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "T-Statistic"

# *****************************************************************************
#
# TTestTwoIndepSamples
#
# *****************************************************************************


@task_decorator("TTestTwoIndepSamples", human_name="T-test two indep. samples result",
                short_description="Test that the means of two independent samples are equal")
class TTestTwoIndepSamples(BasePairwiseStatsTask):
    """
    Compute the T-test for the means of independent samples, from a given reference sample.

    This test is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the T-statistic, and the p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.
    - "equal_variance": a boolean parameter setting whether populations have equal variance.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(TTestTwoIndepSamplesResult,
                                         human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        'equal_variance':
        BoolParam(
            default_value=True, human_name="Equal variance",
            short_description="Set True to assume that the populations have equal variance; False otherwise"),
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=["two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing.")
    }
    _remove_nan_before_compute = False

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        equal_var = params.get_value("equal_variance")
        alternative = params.get_value("alternative_hypothesis")
        stat_result = ttest_ind(*current_data, nan_policy='omit', alternative=alternative, equal_var=equal_var)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result
