# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import ttest_ind
from pandas import DataFrame
import numpy as np

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BoolParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..view.stats_boxplot_view import StatsBoxPlotView
from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("TTestTwoSamplesIndResult", hide=True)
class TTestTwoSamplesIndResult(BaseResource):
    
    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = ['Column 1', 'Column 2', 'T-Statistic', 'p-value']
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

@task_decorator("TTestTwoSamplesInd")
class TTestTwoSamplesInd(Task):
    """
    Compute the T-test for the means of independent samples, from a given reference sample.
    
    This test is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. 

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the T-statistic, and the p-value for each pairwise comparison testing. 
    
    * Config Parameters: 
    - "omit_nan": a boolean parameter setting whether NaN values in the sample measurements are omitted or not. Set True to omit NaN values, False to propagate NaN values
    - "reference_column": the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.
    - "equal_variance": a boolean parameter setting whether populations have equal variance.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html 
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : TTestTwoSamplesIndResult}
    config_specs = {     
        "omit_nan": BoolParam(default_value=True, human_name="Omit NaN", short_description="Set True to omit NaN values, False to propagate NaN values."),
        "reference_column": StrParam(default_value="", human_name="Reference column", short_description="The reference column for pairwise comparison testing. Set empty to use the first column as reference"),
        'equal_variance': BoolParam(default_value=True, human_name="Equal variance", short_description="Set True to assume that the populations have equal variance")
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
        else:
            if array_has_nan:
                self.log_warning_message("Data contain NaN values. NaN values are propagated.")

        col_names = table.column_names
        ref_col = params["reference_column"]
        equal_var = params['equal_variance']
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
                #print(current_data)
                if omit_nan:
                    stat_result = ttest_ind(*current_data, nan_policy='omit', equal_var=equal_var)  
                else:
                    stat_result = ttest_ind(*current_data, nan_policy='propagate', equal_var=equal_var)
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
                #print(current_data)
                if omit_nan:
                    stat_result = ttest_ind(*current_data, nan_policy='omit', equal_var=equal_var)  
                else:
                    stat_result = ttest_ind(*current_data, nan_policy='propagate', equal_var=equal_var)
                stat_result = [ref_col, col_names[i], stat_result.statistic, stat_result.pvalue]
                stat_result = np.array(stat_result)
                all_result = np.vstack((all_result, stat_result))
        
        #-------------------------------
        #suppression de la 1ère ligne artefactuelle qui a servi à initialiser "all_result"
        all_result = np.delete(all_result, 0, 0)        
        #print(all_result)
        #-------------------------------
        result = TTestTwoSamplesIndResult(result = all_result, table=table)
        return {'result': result}
