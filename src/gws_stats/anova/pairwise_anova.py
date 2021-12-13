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

from ..view.stats_boxplot_view import StatsBoxPlotView
from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("AnovaResult", hide=True)
class PairwiseAnovaResult(BaseResource):
    
    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = ['Column 1', 'Column 2', 'F-Statistic', 'p-value']
        data = DataFrame(stat_result, columns=columns)
        return data

    @view(view_type=TableView, human_name="StatTable", short_description="Table of statistic and p-value")
    def view_stats_result_as_table(self, params: ConfigParams) -> TableView:
        """
        View stats Table
        """

        stats_data = self.get_result()
        return TableView(data=stats_data)

    @view(view_type=StatsBoxPlotView, human_name="StatBoxplot", short_description="Boxplot of data with statistic and p-value")
    def view_stats_result_as_boxplot(self, params: ConfigParams) -> StatsBoxPlotView:
        """
        View stats with boxplot of data 
        """

        stats_data = self.get_result()
        return StatsBoxPlotView(self.table, stats=stats_data)

#==============================================================================
#==============================================================================

@task_decorator("ANOVA")
class PairwiseAnova(Task):
    """
    Compute the one-way ANOVA test for pairwise samples, from a given reference sample.
    
    The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.  
    The test is applied to samples from two or more groups, possibly with differing sizes. It is a parametric 
    version of the Kruskal-Wallis test.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the one-way ANOVA F statistic, and the p-value for each pairwise comparison testing. 
    
    * Config Parameters: 
    - "omit_nan": a boolean parameter setting whether NaN values in the sample measurements are omitted or not. Set True to omit NaN values, False to propagate NaN values
    - "reference_column": the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.

    Note: the ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.
    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.
    If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test 
    or the Alexander-Govern test although with some loss of power.
    The length of each group must be at least one, and there must be at least one group with length greater than one.  
    If these conditions are not satisfied, a warning is generated and (``np.nan``, ``np.nan``) is returned.
    If each group contains constant values, and there exist at least two groups with different values, the function 
    generates a warning and returns (``np.inf``, 0).
    If all values in all groups are the same, function generates a warning and returns (``np.nan``, ``np.nan``).
    
    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : PairwiseAnovaResult}
    config_specs = {"reference_column": StrParam(default_value="", human_name="Reference column", short_description="The reference column for pairwise comparison testing. Set empty to use the first column as reference"),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.to_numpy()
        data = data.T

        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            self.log_warning_message("Data contain NaN values.")

        col_names = table.column_names
        ref_col = params["reference_column"]

        #------------------------------
        # construction de la matrice des resultats pour chaque pairwise
        all_result = np.empty([4,])        # initialisation avec données artéfactuelles
        if ref_col == "":
            #--------------------------
            # first column taken as a reference
            ref_col = col_names[0]
            ref_sample = data[0,:]  
            #--------------------------
            for i in range(1, data.shape[0]):
                current_data = [ref_sample, data[i,:]]
                if array_has_nan:
                    #------------------------
                    # removing NaN values from "data"
                    current_data = [[x for x in y if not np.isnan(x)] for y in current_data]
                    #------------------------
                stat_result = f_oneway(*current_data)
                stat_result = [ref_col, col_names[i], stat_result.statistic, stat_result.pvalue]
                stat_result = np.array(stat_result)
                all_result = np.vstack((all_result, stat_result))
        else:
            #--------------------------
            # "ref_col" taken as reference column
            nb_ref_col = col_names.index(ref_col)
            ref_sample = data[nb_ref_col,:]
            #--------------------------
            indeces = [i for i in range(data.shape[0])]
            indeces.pop(nb_ref_col)
            for i in indeces:
                current_data = [ref_sample, data[i,:]]
                if array_has_nan:
                    #------------------------
                    # removing NaN values from "data"
                    current_data = [[x for x in y if not np.isnan(x)] for y in current_data]
                    #------------------------                
                stat_result = f_oneway(*current_data)  
                stat_result = [ref_col, col_names[i], stat_result.statistic, stat_result.pvalue]
                stat_result = np.array(stat_result)
                all_result = np.vstack((all_result, stat_result))
        
        #-------------------------------
        #suppression de la 1ère ligne artefactuelle qui a servi à initialiser "all_result"
        all_result = np.delete(all_result, 0, 0)        
        #-------------------------------
        result = PairwiseAnovaResult(result = all_result, table=table)        
        #print(all_result)
        return {'result': result}


