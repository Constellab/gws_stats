
import numpy as np
from gws_core import (ConfigParams, InputSpec, InputSpecs, OutputSpec,
                      OutputSpecs, StrParam, Table, resource_decorator,
                      task_decorator)
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
      - `column_names`: The columns used for pairwise comparison. By default, the first three columns are used.
      - `method`: the method used to calculate the p-value (either `auto`, `asymptotic`, or `exact`). Default method is set to `auto`.
      - `alternative_hypothesis`: the alternative hypothesis chosen for the testing (either `two-sided`, `less`, or `greater`). Default alternative hypothesis is set to "two-sided".

    Note: the `exact` method is recommended when there are no ties and when either sample size is less than 8.
    The `exact` method is not corrected for ties, but no errors or warnings will be raised if there are ties in the data.
    The Mann-Whitney U test is a non-parametric version of the t-test for independent samples. When the the means of samples
    from the populations are normally distributed, consider the t-test for independant samples.
    Note that the Mann-Whitney U statistic depends on the sample take as the first one for the computation of the statistic

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
      - `method`: Method used to calculate teh p-value (`auto`, `asymptotic` or `exact`)
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

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    """
    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(
        MannWhitneyResult, human_name="Result", short_description="The output result")})
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
        stat_result = mannwhitneyu(
            *current_data, method=method, alternative=alternative)
        stat_result = [ref_col, target_col,
                       stat_result.statistic, stat_result.pvalue]
        return stat_result
