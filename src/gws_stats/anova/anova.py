# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, InputSpec, ListParam, OutputSpec, Table,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view)
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

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: the one-way ANOVA F statistic, and the corresponding p-value.
    * Config Parameters:

    The ANOVA has important assumptions that must be satisfied in order for the associated p-value to be valid:
    * The samples are independent.
    * Each sample is from a normally distributed population.
    * The population standard deviations of the groups are all equal.This property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test
    or the Alexander-Govern test although with some loss of power.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(OneWayAnovaResult, human_name="Result", short_description="The output result")}

    def compute_stats(self, data, _: ConfigParams):
        """ Compute stats """
        stat_result = f_oneway(*data)
        return stat_result
