# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import mannwhitneyu
from pandas import DataFrame
import numpy as np

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..view.stats_boxplot_view import StatsBoxPlotView
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("MannWhitneyResult", hide=True)
class MannWhitneyResult(BaseResource):
    
    def get_result(self) -> DataFrame:
        stat_result = super().get_result()
        columns = ['Column 1', 'Column 2', 'U-Statistic', 'p-value']
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

@task_decorator("MannWhitney")
class MannWhitney(Task):
    """
    Mann Whitney U rank test on pairwise independent samples, from a given sample reference.
    
    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distributions underlying 
    two samples are the same. It is often used to test whether two samples are likely to derive from the same population

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the Mann-Whitney U statistic, and the associated p-value for each pairwise comparison testing. 

    * Config Parameters:
    - "reference_column": the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.
    - "method": the method used to calculate the p-value (either "auto", "asymptotic", or "exact"). Default method is set to "auto".
    - "alternative_hypothesis": the alternative hypothesis chosen for the testing (either "two-sided", "less", or "greater"). Default alternative hypothesis is set to "two-sided".

    Note: the "exact" method is recommended when there are no ties and when either sample size is less than 8. 
    The "exact" method is not corrected for ties, but no errors or warnings will be raised if there are ties in the data.
    The Mann-Whitney U test is a non-parametric version of the t-test for independent samples. When the the means of samples 
    from the populations are normally distributed, consider the t-test for independant samples.
    Note that the Mann-Whitney U statistic depends on the sample take as the first one for the computation of the statistic

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html 
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : MannWhitneyResult}
    config_specs = {
        "method": StrParam(default_value="auto", human_name="Method for p-value computation", short_description="Method used to calculate teh p-value"),
        "reference_column": StrParam(default_value="", human_name="Reference column", short_description="The reference column for pairwise comparison testing. Set empty to use the first column as reference"),
        "alternative_hypothesis": StrParam(default_value="two-sided", human_name="Alternative hypothesis", short_description="The alternative hypothesis chosen for the testing.")
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
        method = params["method"]
        alternat = params["alternative_hypothesis"]

        #------------------------------
        # construction de la matrice des resultats pour chaque pairwise
        all_result = np.empty([data.shape[1],])        # initialisation avec données artéfactuelles
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
                stat_result = mannwhitneyu(*current_data, method=method, alternative = alternat)
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
                stat_result = mannwhitneyu(*current_data, method=method, alternative = alternat)  
                stat_result = [ref_col, col_names[i], stat_result.statistic, stat_result.pvalue]
                stat_result = np.array(stat_result)
                all_result = np.vstack((all_result, stat_result))
        
        #-------------------------------
        #suppression de la 1ère ligne artefactuelle qui a servi à initialiser "all_result"
        all_result = np.delete(all_result, 0, 0)        
        #-------------------------------
        result = MannWhitneyResult(result = all_result, table=table)        
        #print(all_result)
        return {'result': result}

