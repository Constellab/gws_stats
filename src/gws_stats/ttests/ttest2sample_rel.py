# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (BoolParam, ConfigParams, InputSpec, OutputSpec, StrParam,
                      Table, resource_decorator, task_decorator)
from scipy.stats import ttest_rel

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# TTestTwoRelatedSamplesResult
#
# *****************************************************************************


@resource_decorator("TTestTwoRelatedSamplesResult", human_name="T-test with rel. samples result",
                    short_description="Result of related samples Student test(T-Test)", hide=True)
class TTestTwoRelatedSamplesResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "TStatistic"

# *****************************************************************************
#
# TTestTwoRelatedSamples
#
# *****************************************************************************


@ task_decorator("TTestTwoRelatedSamples", human_name="T-test with rel. samples",
                 short_description="Test that the means of two related samples are equal. Performs pairwise analysis for more than two samples.")
class TTestTwoRelatedSamples(BasePairwiseStatsTask):
    """
    Compute the T-test for the means of related samples, from a given reference sample.

    This test is a two-sided (or one-side) test for the null hypothesis that 2 related samples have identical average (expected) values.
    Performs pairwise analysis for more than two samples.

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the T-statistic, and the p-value for each pairwise comparison testing.
    * Config Parameters:
      - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(TTestTwoRelatedSamplesResult,
                                         human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=["two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing.")
    }
    _remove_nan_before_compute = True # ensure that related sample are paired!

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        alternative = params.get_value("alternative_hypothesis")
        stat_result = ttest_rel(*current_data, nan_policy='omit', alternative=alternative)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result
