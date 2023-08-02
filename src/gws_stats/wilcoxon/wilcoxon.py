# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoolParam, ConfigParams, InputSpec, InputSpecs,
                      OutputSpec, OutputSpecs, StrParam, Table,
                      resource_decorator, task_decorator)
from scipy.stats import wilcoxon

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# ==============================================================================
# ==============================================================================


@resource_decorator("WilcoxonResult", human_name="Wilcoxon test result",
                    short_description="Result of Wilcoxon test", hide=True)
class WilcoxonResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "TStatistic"

# ==============================================================================
# ==============================================================================


@task_decorator("Wilcoxon", human_name="Paired Wilcoxon test",
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
      - `reference_column`: the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.
      - `zero_method`: the method used to treat the zero differences. The following options are available (default is “wilcox”):
        - `pratt`: Includes zero-differences in the ranking process, but drops the ranks of the zeros, see [4], (more conservative).
        - `wilcox`: Discards all zero-differences, the default.
        - `zsplit`: Includes zero-differences in the ranking process and split the zero rank between positive and negative ones.
      - `alternative_hypothesis`: the alternative hypothesis to be tested (either "less, "greater", or "two_sided"). Default is “two-sided”.
      - `mode`:  the method to calculate the p-value (either ""auto", "exact", or "approx"). Default is "auto".

    Notes: one assumption of the test is that the differences are symmetric. The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from zero. The one-sided test has the null hypothesis that the median is
    positive against the alternative that it is negative, or vice versa.
    To derive the p-value, the exact distribution ('mode' == 'exact') can be used for sample sizes of up to 25. The default "mode" == "auto"
    uses the exact distribution if there are at most 25 observations and no ties, otherwise a normal approximation is used ("mode" == "approx").
    The treatment of ties can be controlled by the parameter "zero_method".

    ---

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.
    * Config Parameters:
      - `preselected_column_names`: List of columns to pre-select for pairwise comparisons. By default a maximum pre-defined number of columns are selected (see configuration).
      - `reference_column`: If given, this reference column is compared against all the other columns.
      - `row_tag_key`: If give, this parameter is used for group-wise comparisons along row tags (see example below). This parameter is ignored of a `reference_column` is given.
      - `adjust_pvalue`:
        - `method`: The correction method for p-value adjustment in multiple testing.
        - `alpha`: The FWER, family-wise error rate. Default is 0.05.

    # Example 1: Direct column comparisons

    Let's say you have the following table.

    | A | B | C |
    |---|---|---|
    | 1 | 5 | 3 |
    | 2 | 6 | 8 |
    | 3 | 7 | 5 |
    | 4 | 8 | 4 |

    This task performs pairwise comparison of almost all the columns of the table
    (the first `500` columns are pre-selected by default).

    - `A` will be compared with `B` and with `C`, respectively
    - `B` will be compared with `C`

    To only compare a given column with all the others, set the name of the `reference_column` (a.k.a Reference column).
    Suppose that `B` is used as reference column,only the following comaprisons will be done:

    - `B` versus `A`
    - `B` versus `C`

    It is also possible to perform comparison on a well-defined subset of the table by pre-selecting the columns of interest.
    Parameter `preselected_column_names` (a.k.a. Selected columns names) allows pre-selecting a subset of columns for analysis.

    # Example 2: Advanced comparisons along row tags using `row_tag_key` parameter

    In general, the table rows represent real-world observations (e.g. measured samples) and columns correspond to
    descriptors (a.k.a features or variables).
    Theses rows (samples) may therefore be related to metadata information given by row tags as follows:

    | row_tags                 | A | B | C |
    |--------------------------|---|---|---|
    | Gender : M <br> Age : 10 | 1 | 5 | 3 |
    | Gender : F <br> Age : 10 | 2 | 6 | 8 |
    | Gender : F <br> Age : 10 | 3 | 7 | 5 |
    | Gender : M <br> Age : 20 | 4 | 8 | 4 |

    Actually, the column ```row_tags``` does not really exist in the table. It is just to show here the tags of the rows
    Here, the first row correspond to 10-years old male individuals.
    In this this case, we may be interested in only comparing each columns along row metadata tags.
    For instance, to compare `Males (M)` versus `Females (F)` of each columns separately, you can use the advance parameter `row_tag_key`=`Gender`.

    ---

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(
        WilcoxonResult, human_name="Result", short_description="The output result")})
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
