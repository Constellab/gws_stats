# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import kruskal
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KruskalWallisResult", hide=True)
class KruskalWallisResult(BaseResource):
    
    @view(view_type=TableView, human_name="VarianceTable", short_description="Table of explained variances")
    def view_scores_as_table(self, params: ConfigParams) -> dict:
        """
        View score Table
        """

        # pca = self.get_result()
        # index = [f"PC{n+1}" for n in range(0,pca.n_components_)]
        # columns = ["ExplainedVariance"]
        # data = DataFrame(pca.explained_variance_ratio_, index=index, columns=columns)
        return TableView(data=data)

#==============================================================================
#==============================================================================

@task_decorator("KruskalWallis")
class KruskalWallis(Task):
    """
    Test statistique de Kruskal Wallis
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : DataFrame}
    config_specs = {
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']

        
        # neigh = KNeighborsClassifier(n_neighbors=params["nb_neighbors"])
        # neigh.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        # result = KNNClassifierResult(result = neigh)

        return {'result': result}



