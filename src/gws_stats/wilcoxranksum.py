# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import ranksums
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource


#==============================================================================
#==============================================================================

@resource_decorator("WilcoxRankSumResult", hide=True)
class WilcoxRankSumResult(BaseResource):
    pass
#==============================================================================
#==============================================================================

@task_decorator("WilcoxRankSum")
class WilcoxRankSum(Task):
    """
    Wilcoxon rank sum test
    """
    input_specs = {'dataset' : DataFrame}
    output_specs = {'result' : DataFrame}
    config_specs = {

    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']

        # neigh = KNeighborsClassifier(n_neighbors=params["nb_neighbors"])
        # neigh.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        # result = KNNClassifierResult(result = neigh)

        return {'result': result}

