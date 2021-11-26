# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import ttest_rel
from pandas import DataFrame
from numpy import array

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BoolParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("TTestTwoSamplesPairedResult", hide=True)
class TTestTwoSamplesPairedResult(BaseResource):
    
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

@task_decorator("TTestTwoSamplesPaired")
class TTestTwoSamplesPaired(Task):
    """
    T test sur la moyenne de 2 échantillons liés

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html 
    """
    input_specs = {'dataset' : Table}
    output_specs = {'result' : TTestTwoSamplesPairedResult}
    config_specs = {        
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']

        data = dataset.get_data()
        data = data.to_numpy()
        data = data.T
        stat_result = ttest_rel(*data, nan_policy='omit')            
        
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = array(stat_result)
        result = TTestTwoSamplesPairedResult(result = stat_result)
        return {'result': result}



