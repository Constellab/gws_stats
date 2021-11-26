# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import f_oneway
from pandas import DataFrame
from numpy import array

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("AnovaResult", hide=True)
class AnovaResult(BaseResource):
    
    @view(view_type=TableView, human_name="ScoresTable", short_description="Table of scores")
    def view_scores_as_table(self, params: ConfigParams) -> dict:
        """
        View score Table
        """

        stat_result = self.get_result()
        columns = ['Statistic', 'p-value']
        data = DataFrame([stat_result], columns=columns)
        return TableView(data=data)

#==============================================================================
#==============================================================================

@task_decorator("ANOVA")
class Anova(Task):
    """
    Test statistique ANOVA
    """
    input_specs = {'dataset' : Table}
    output_specs = {'result' :AnovaResult}
    config_specs = {
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        data = dataset.get_data()
        data = data.to_numpy()
        data = data.T
        #------------------------
        # removing NaN values from "data"
        data = [[x for x in y if not isnan(x)] for y in data]
        #------------------------

        stat_result = f_oneway(*data)            
        
        stat_result = [stat_result.statistic, stat_result.pvalue]
        stat_result = array(stat_result)
        result = AnovaResult(result = stat_result)
        return {'result': result}



