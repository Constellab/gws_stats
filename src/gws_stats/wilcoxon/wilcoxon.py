# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoolParam, ConfigParams, InputSpec, OutputSpec, StrParam,
                      Table, resource_decorator, task_decorator)
from scipy.stats import wilcoxon

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# ==============================================================================
# ==============================================================================


@resource_decorator("WilcoxonResult", human_name="Wilcoxon test result",
                    short_description="Result of Wilcoxon test", hide=True)
class WilcoxonResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "T-Statistic"

# ==============================================================================
# ==============================================================================


@task_decorator("Wilcoxon", human_name="Wilcoxon test",
                short_description="Test that two related paired samples come from the same distribution")
class Wilcoxon(BasePairwiseStatsTask):
    """
    Calculate the Wilcoxon signed-rank test of paired samples, from a given reference sample.

    The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same
    distribution. In particular, it tests whether the distribution of the differences between the two samples is symmetric about zero.
    It is a non-parametric version of the paired T-test.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the Wilcoxon T-statistic, and the p-value for each pairwise comparison testing.

    * Config Parameters:
    - "reference_column": the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.
    - "zero_method": the method used to treat the zero differences. The following options are available (default is “wilcox”):
            “pratt”: Includes zero-differences in the ranking process, but drops the ranks of the zeros, see [4], (more conservative).
            “wilcox”: Discards all zero-differences, the default.
            “zsplit”: Includes zero-differences in the ranking process and split the zero rank between positive and negative ones.
    - "alternative_hypothesis": the alternative hypothesis to be tested (either "less, "greater", or "two_sided"). Default is “two-sided”.
    - "mode":  the method to calculate the p-value (either ""auto", "exact", or "approx"). Default is "auto".

    Notes: one assumption of the test is that the differences are symmetric. The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from zero. The one-sided test has the null hypothesis that the median is
    positive against the alternative that it is negative, or vice versa.
    To derive the p-value, the exact distribution ('mode' == 'exact') can be used for sample sizes of up to 25. The default "mode" == "auto"
    uses the exact distribution if there are at most 25 observations and no ties, otherwise a normal approximation is used ("mode" == "approx").
    The treatment of ties can be controlled by the parameter "zero_method".

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(WilcoxonResult, human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        "zero_method": StrParam(default_value="wilcox",
                                allowed_values=["pratt", "wilcox", "zsplit"],
                                human_name="Method for zero differences treatment",
                                short_description="Method chosen to include or not zero differences and their ranking"),
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=["two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing."),
        "mode": StrParam(default_value="auto",
                         allowed_values=["auto", "exact", "approx"],
                         human_name="Mode",
                         short_description="Method to calculate the p-value.")
    }

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        mode = params.get_value("mode")
        zero_method = params.get_value("zero_method")
        alternative = params.get_value("alternative_hypothesis")
        stat_result = wilcoxon(*current_data, zero_method=zero_method, alternative=alternative, mode=mode)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result
