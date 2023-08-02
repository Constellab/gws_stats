# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, InputSpec, InputSpecs, ListParam,
                      OutputSpec, OutputSpecs, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from scipy.stats import kruskal

from ..base.base_population_stats_result import BasePopulationStatsResult
from ..base.base_population_stats_task import BasePopulationStatsTask

# *****************************************************************************
#
# KruskalWallisResult
#
# *****************************************************************************


@resource_decorator("KruskalWallisResult", human_name="Population Kruskal-Wallis result",
                    short_description="Result of multiple Kruskal-Wallis H-test", hide=True)
class KruskalWallisResult(BasePopulationStatsResult):
    """ KruskalWallisResult """
    STATISTICS_NAME = "H-Statistic"

# *****************************************************************************
#
# KruskalWallis
#
# *****************************************************************************


@task_decorator("KruskalWallis", human_name="Kruskal-Wallis",
                short_description="Test that two or more groups have the same population median")
class KruskalWallis(BasePopulationStatsTask):
    """
    Compute the Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the population
    median of all of the groups are equal.  It is a non-parametric version of
    ANOVA.  The test works on 2 or more independent samples, which may have
    different sizes.  Note that rejecting the null hypothesis does not
    indicate which of the groups differs.  Post hoc comparisons between
    groups are required to determine which groups are different.

    Note: due to the assumption that H has a chi square distribution, the number
    of samples in each group must not be too small. A typical rule is that each sample
    must have at least 5 measurements.

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.
    * Config Parameters:
      - `preselected_column_names`: List of columns to pre-select for pairwise comparisons. By default a maximum pre-defined number of columns are selected (see configuration).
      - `row_tag_key`: If give, this parameter is used for group-wise comparisons along row tags (see example below).

    ---

    # Example 1: Direct column comparisons

    Let's say you have the following table.

    | A | B | C |
    |---|---|---|
    | 1 | 5 | 3 |
    | 2 | 6 | 8 |
    | 3 | 7 | 5 |
    | 4 | 8 | 4 |

    This task performs population comparison of almost all the columns of the table
    (the first `500` columns are pre-selected by default).

    # Example 2: Advanced comparisons along row tags using `row_tag_key` parameter

    In general, the table rows represent real-world observations (e.g. measured samples) and columns correspond to
    descriptors (a.k.a features or variables).
    Theses rows (samples) may therefore be related to metadata information given by row tags as follows:

    | row_tags                 | A | B | C |
    |--------------------------|---|---|---|
    | Gender : M <br> Age : 10 | 1 | 5 | 3 |
    | Gender : F <br> Age : 10 | 2 | 6 | 8 |
    | Gender : F <br> Age : 10 | 8 | 7 | 5 |
    | Gender : X <br> Age : 20 | 4 | 8 | 4 |
    | Gender : X <br> Age : 10 | 2 | 7 | 5 |
    | Gender : M <br> Age : 20 | 4 | 1 | 4 |

    Actually, the column ```row_tags``` does not really exist in the table. It is just to show here the tags of the rows
    Here, the first row correspond to 10-years old male individuals.
    In this this case, we may be interested in only comparing several columns along row metadata tags.
    For instance, to compare gender populations `M`, `F`, `X` for each columns separately, you can therefore use the advance parameter `row_tag_key`=`Gender`.

    ---

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    """

    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(KruskalWallisResult, human_name="Result",
                                                     short_description="The output result")})

    def compute_stats(self, data, _: ConfigParams):
        """ Compute stats """
        stat_result = kruskal(*data)
        return stat_result
