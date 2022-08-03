# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, InputSpec, OutputSpec, StrParam, Table,
                      resource_decorator, task_decorator)
from scipy.stats import mannwhitneyu

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# MannWhitneyResult
#
# *****************************************************************************


@resource_decorator("MannWhitneyResult", human_name="Mann Whitney result",
                    short_description="Result of pairwise Mann Whitney U-rank test", hide=True)
class MannWhitneyResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "U-Statistic"

# *****************************************************************************
#
# MannWhitney
#
# *****************************************************************************


@task_decorator("MannWhitney", human_name="Mann Whitney",
                short_description="Test that the distributions of two samples are the same")
class MannWhitney(BasePairwiseStatsTask):
    """
    Mann Whitney U rank test on pairwise independent samples, from a given sample reference.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distributions underlying
    two samples are the same. It is often used to test whether two samples are likely to derive from the same population

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the Mann-Whitney U statistic, and the associated p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.
    - "method": the method used to calculate the p-value (either "auto", "asymptotic", or "exact"). Default method is set to "auto".
    - "alternative_hypothesis": the alternative hypothesis chosen for the testing (either "two-sided", "less", or "greater"). Default alternative hypothesis is set to "two-sided".

    Note: the "exact" method is recommended when there are no ties and when either sample size is less than 8.
    The "exact" method is not corrected for ties, but no errors or warnings will be raised if there are ties in the data.
    The Mann-Whitney U test is a non-parametric version of the t-test for independent samples. When the the means of samples
    from the populations are normally distributed, consider the t-test for independant samples.
    Note that the Mann-Whitney U statistic depends on the sample take as the first one for the computation of the statistic

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(MannWhitneyResult, human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        "method":
        StrParam(
            default_value="auto", human_name="Method for p-value computation",
            allowed_values=["auto", "asymptotic", "exact"],
            short_description="Method used to calculate teh p-value"),
        "alternative_hypothesis":
        StrParam(
            default_value="two-sided",
            allowed_values=["two-sided", "less", "greater"],
            human_name="Alternative hypothesis",
            short_description="The alternative hypothesis chosen for the testing.")}

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        method = params.get_value("method")
        alternative = params.get_value("alternative_hypothesis")
        stat_result = mannwhitneyu(*current_data, method=method, alternative=alternative)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result
