
import numpy as np
from gws_core import (ConfigParams, InputSpec, InputSpecs, OutputSpec,
                      OutputSpecs, Table, resource_decorator, task_decorator)
from scipy.stats import kruskal

from ...base.base_pairwise_stats_result import BasePairwiseStatsResult
from ...base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# PairwiseKruskalWallisResult
#
# *****************************************************************************


@resource_decorator("PairwiseKruskalWallisResult", human_name="Pairwise Kruskal-Wallis result",
                    short_description="Result of pairwise Kruskal-Wallis H-test", hide=True,
                    deprecated_since='0.3.1', deprecated_message="This resource is deprecated")
class PairwiseKruskalWallisResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "H-Statistic"

# *****************************************************************************
#
# PairwiseKruskalWallis
#
# *****************************************************************************


@task_decorator("PairwiseKruskalWallis", human_name="Pairwise Kruskal-Wallis",
                short_description="Test that two groups have the same population median",
                hide=True, deprecated_since='0.3.1', deprecated_message="This task is deprecated")
class PairwiseKruskalWallis(BasePairwiseStatsTask):
    """
    Compute the Kruskal-Wallis H-test for pairwise independent samples, from a given reference sample.

    The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal.
    It is a non-parametric version of ANOVA.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the Kruskal-Wallis H statistic, corrected for ties, and the p-value for each pairwise comparison testing.
    Kruskal-Wallis H statistic is corrected for ties. The p-value for the test uses the assumption that H has a chi square distribution. The p-value returned is the survival function of the chi square distribution evaluated at H.

     for the test using the assumption that H has a chi
       square distribution. The p-value returned is the survival function of the chi square distribution evaluated at H.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    Note: due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small.  A typical rule is
    that each sample must have at least 5 measurements.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    """
    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(PairwiseKruskalWallisResult,
                                                     human_name="Result", short_description="The output result")})
    config_specs = {
        **BasePairwiseStatsTask.config_specs
    }
    _remove_nan_before_compute = False

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        stat_result = kruskal(*current_data, nan_policy='omit')
        stat_result = [ref_col, target_col,
                       stat_result.statistic, stat_result.pvalue]
        return stat_result
