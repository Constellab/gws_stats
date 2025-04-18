
import numpy as np
from gws_core import (ConfigParams, ConfigSpecs, InputSpec, InputSpecs,
                      OutputSpec, OutputSpecs, StrParam, Table,
                      resource_decorator, task_decorator)
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


@task_decorator("TTestTwoRelatedSamples", human_name="T-test with rel. samples",
                short_description="Test that the means of two related samples are equal. Performs pairwise analysis for more than two samples.")
class TTestTwoRelatedSamples(BasePairwiseStatsTask):
    """
    Compute the T-test for the means of related samples, from a given reference sample.

    This test is a two-sided (or one-side) test for the null hypothesis that 2 related samples have identical average (expected) values.
    Performs pairwise analysis for more than two samples.

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
     - `alternative_hypothesis`: The alternative hypothesis chosen for the testing (`two-sided`, `less` or `greater`)

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


    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(TTestTwoRelatedSamplesResult,
                                                     human_name="Result", short_description="The output result")})
    config_specs = ConfigSpecs({
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=[
                                               "two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing")
    }).merge_specs(BasePairwiseStatsTask.config_specs)
    _remove_nan_before_compute = True  # ensure that related sample are paired!

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        alternative = params.get_value("alternative_hypothesis")
        stat_result = ttest_rel(
            *current_data, nan_policy='omit', alternative=alternative)
        stat_result = [ref_col, target_col,
                       stat_result.statistic, stat_result.pvalue]
        return stat_result
