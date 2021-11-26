# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import ttest_1samp
from pandas import DataFrame
from numpy import array

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("TTestOneSampleResult", hide=True)
class TTestOneSampleResult(BaseResource):
    
    @view(view_type=TableView, human_name="ScoresTable", short_description="Table of scores")
    def view_scores_as_table(self, params: ConfigParams) -> dict:
        """
        View score Table
        """

        stat_result = self.get_result()
        columns = ['T statistic', 'p-value']
        data = DataFrame([stat_result], columns=columns)
        return TableView(data=data)

#==============================================================================
#==============================================================================

@task_decorator("TtestOneSample")
class TTestOneSample(Task):
    """
    T test pour la moyenne d'un échantillon

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    """
    input_specs = {'dataset' : Table}
    output_specs = {'result' : TTestOneSampleResult}
    config_specs = {        
        'expected_value': FloatParam(default_value=0) 
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        exp_val = params["expected_value"]

        data = dataset.get_data()
        data = data.to_numpy()
        data = data.T
        stat_result = ttest_1samp(*data, exp_val, nan_policy='omit')            
        
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = array(stat_result)
        result = TTestOneSampleResult(result = stat_result)
        return {'result': result}



