# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator)
from scipy.stats import f_oneway

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# PairwiseOneWayAnovaResult
#
# *****************************************************************************


@resource_decorator("PairwiseOneWayAnovaResult", human_name="Result of pairwise one-way ANOVA", hide=True)
class PairwiseOneWayAnovaResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "F-Statistic"

# *****************************************************************************
#
# PairwiseOneWayAnova
#
# *****************************************************************************


@task_decorator("PairwiseOneWayAnova", human_name="Pairwise one-way ANOVA",
                short_description="Test that two groups have the same population mean")
class PairwiseOneWayAnova(BasePairwiseStatsTask):
    """
    Compute the one-way ANOVA test for pairwise samples, from a given reference sample.

    The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
    The test is applied to samples from two or more groups, possibly with differing sizes. It is a parametric
    version of the Kruskal-Wallis test.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the one-way ANOVA F statistic, and the p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    Note: the ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test
    or the Alexander-Govern test although with some loss of power.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(PairwiseOneWayAnovaResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs
    }

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        stat_result = f_oneway(*current_data)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result
