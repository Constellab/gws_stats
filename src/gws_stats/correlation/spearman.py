# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, InputSpec, InputSpecs, OutputSpec,
                      OutputSpecs, StrParam, Table, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from scipy.stats import spearmanr

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# SpearmanCorrelationResult
#
# *****************************************************************************


@resource_decorator("SpearmanCorrelationResult",
                    human_name="Result of pairwise spearman correlation coefficient analysis", hide=True)
class SpearmanCorrelationResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "Correlation"

# *****************************************************************************
#
# SpearmanCorrelation
#
# *****************************************************************************


@task_decorator("SpearmanCorrelation", human_name="Spearman correlation",
                short_description="Compute Spearman correlation coefficients between two groups with p-value")
class SpearmanCorrelation(BasePairwiseStatsTask):
    """
    Compute the Spearman correlation coefficient for pairwise samples, with its p-value.

    The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship
    between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are
    normally distributed. The p-value returned is a two-sided p-value.
    Like other correlation coefficients, these ones vary between -1 and +1 with 0 implying no correlation. Correlations of -1
    or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations
    imply that as x increases, y decreases.

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

    For more details on the Spearman correlation coefficient, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.htm.
    """

    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(SpearmanCorrelationResult, human_name="Result",
                                                     short_description="The output result")})
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
    }

    _remove_nan_before_compute = False

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        # remove nan values and clean data to have same column lengths
        idx = ~np.isnan(current_data).any(axis=0)
        current_data = current_data[:, idx]
        # compute stats
        try:
            stat_result = spearmanr(current_data[0], current_data[1])
            stat_result = [ref_col, target_col, stat_result[0], stat_result[1]]
        except Exception as _:
            stat_result = [ref_col, target_col, np.nan, np.nan]
        return stat_result
