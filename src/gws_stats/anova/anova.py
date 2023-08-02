# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, InputSpec, InputSpecs, OutputSpec,
                      OutputSpecs, Table, resource_decorator, task_decorator)
from scipy.stats import f_oneway

from ..base.base_population_stats_result import BasePopulationStatsResult
from ..base.base_population_stats_task import BasePopulationStatsTask

# *****************************************************************************
#
# AnovaResult
#
# *****************************************************************************


@resource_decorator("OneWayAnovaResult", human_name="One-way ANOVA result",
                    short_description="Result of one-way ANOVA test for multiple samples", hide=True)
class OneWayAnovaResult(BasePopulationStatsResult):
    """ OneWayAnovaResult """
    STATISTICS_NAME = "F-Statistic"

# *****************************************************************************
#
# OneWayAnova
#
# *****************************************************************************


@task_decorator("OneWayAnova", human_name="One-way ANOVA",
                short_description="Test that two or more groups have the same population mean")
class OneWayAnova(BasePopulationStatsTask):
    """
    Compute the one-way ANOVA test for multiple samples.

    The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
    The test is applied to samples from two or more groups, possibly with differing sizes. It is a parametric
    version of the Kruskal-Wallis test.

    The ANOVA has important assumptions that must be satisfied in order for the associated p-value to be valid:
    * The samples are independent.
    * Each sample is from a normally distributed population.
    * The population standard deviations of the groups are all equal.This property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test
    or the Alexander-Govern test although with some loss of power.

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

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """

    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(OneWayAnovaResult, human_name="Result", short_description="The output result")})

    def compute_stats(self, data, _: ConfigParams):
        """ Compute stats """
        stat_result = f_oneway(*data)
        return stat_result
