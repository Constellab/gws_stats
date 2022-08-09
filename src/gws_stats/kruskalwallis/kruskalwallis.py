# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, InputSpec, ListParam, OutputSpec, Table,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator)
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

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: the Kruskal-Wallis H statistic, corrected for ties, and the p-value for the test using the assumption that H has a chi
       square distribution. The p-value returned is the survival function of the chi square distribution evaluated at H.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    Note: due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small.  A typical rule is
    that each sample must have at least 5 measurements.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    """

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(KruskalWallisResult, human_name="Result",
                                         short_description="The output result")}

    def compute_stats(self, data, _: ConfigParams):
        """ Compute stats """
        stat_result = kruskal(*data)
        return stat_result
