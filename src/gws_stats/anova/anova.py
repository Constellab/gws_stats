# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import f_oneway
from pandas import DataFrame
import numpy as np

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BoxPlotView,
                        StrParam, BoolParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("AnovaResult", hide=True)
class AnovaResult(BaseResource):

    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = ['F-Statistic', 'p-value']
        data = DataFrame([stat_result], columns=columns)
        return data
    
    @view(view_type=TableView, human_name="StatTable", short_description="Table of statistic and p-value")
    def view_stats_result_as_table(self, params: ConfigParams) -> dict:
        """
        View stats Table
        """

        stat_result = self.get_result()
        return TableView(data=stat_result)

#==============================================================================
#==============================================================================

@task_decorator("Anova")
class Anova(Task):
    """
    Compute the one-way ANOVA test for multiple samples.
    
    The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.  
    The test is applied to samples from two or more groups, possibly with differing sizes. It is a parametric 
    version of the Kruskal-Wallis test.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: the one-way ANOVA F statistic, and the corresponding p-value. 
    
    * Config Parameters: 

    Note: the ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test 
    or the Alexander-Govern test although with some loss of power.
    
    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : AnovaResult}
    config_specs = { 
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.to_numpy()
        data = data.T

        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)

        if array_has_nan:
            self.log_warning_message("Data contain NaN values. NaN values are omitted.")
            #-------------------------------------------
            # removing NaN values from "data"
            data = [[x for x in y if not np.isnan(x)] for y in data]
            #-------------------------------------------

        stat_result = f_oneway(*data)    
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        result = AnovaResult(result = stat_result, table=table)
        return {'result': result}



