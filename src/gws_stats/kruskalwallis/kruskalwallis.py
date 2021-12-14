# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import kruskal
from pandas import DataFrame
import numpy as np

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BoxPlotView,
                        StrParam, BoolParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("KruskalWallisResult", hide=True)
class KruskalWallisResult(BaseResource):

    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = ['H-Statistic', 'p-value']
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

@task_decorator("KruskalWallis")
class KruskalWallis(Task):
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
    - "omit_nan": a boolean parameter setting whether NaN values in the sample measurements are omitted or not. Set True to omit NaN values, False to propagate NaN values 

    Note: due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small.  A typical rule is 
    that each sample must have at least 5 measurements.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html 
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : KruskalWallisResult}
    config_specs = { 
        "omit_nan": BoolParam(default_value=True, human_name="Omit NaN", short_description="Set True to omit NaN values, False to propagate NaN values.")
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.to_numpy()
        data = data.T

        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)
        omit_nan = params["omit_nan"]

        if omit_nan:
            if array_has_nan:
                self.log_warning_message("Data contain NaN values. NaN values are omitted.")
            stat_result = kruskal(*data, nan_policy='omit')  
        else:
            if array_has_nan:
                self.log_warning_message("Data contain NaN values. NaN values are propagated.")
            stat_result = kruskal(*data, nan_policy='propagate')
        
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        result = KruskalWallisResult(result = stat_result, table=table)
        return {'result': result}



