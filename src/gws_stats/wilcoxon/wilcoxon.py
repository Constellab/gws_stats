# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from scipy.stats import wilcoxon
from pandas import DataFrame
import numpy as np

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BoolParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource, Table)

from ..view.stats_boxplot_view import StatsBoxPlotView
from ..base.base_resource import BaseResource
#==============================================================================
#==============================================================================

@resource_decorator("WilcoxonResult", hide=True)
class WilcoxonResult(BaseResource):
    
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

@task_decorator("Wilcoxon")
class Wilcoxon(Task):
    """
    Calculate the Wilcoxon signed-rank test of paired samples, from a given reference sample.

    The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same 
    distribution. In particular, it tests whether the distribution of the differences between the two samples is symmetric about zero. 
    It is a non-parametric version of the paired T-test.
    
    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the Wilcoxon T-statistic, and the p-value for each pairwise comparison testing. 
    
    * Config Parameters: 
    - "reference_column": the name of the reference sample for pairwise comparison testing. Set it to empty to use the first column of the table of samples as reference.
    - "zero_method": the method used to treat the zero differences. The following options are available (default is “wilcox”):
            “pratt”: Includes zero-differences in the ranking process, but drops the ranks of the zeros, see [4], (more conservative).
            “wilcox”: Discards all zero-differences, the default.
            “zsplit”: Includes zero-differences in the ranking process and split the zero rank between positive and negative ones.
    - "alternative_hypothesis": the alternative hypothesis to be tested (either "less, "greater", or "two_sided"). Default is “two-sided”.
    - "mode":  the method to calculate the p-value (either ""auto", "exact", or "approx"). Default is "auto".

    Notes: one assumption of the test is that the differences are symmetric. The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from zero. The one-sided test has the null hypothesis that the median is
    positive against the alternative that it is negative, or vice versa.
    To derive the p-value, the exact distribution ('mode' == 'exact') can be used for sample sizes of up to 25. The default "mode" == "auto"
    uses the exact distribution if there are at most 25 observations and no ties, otherwise a normal approximation is used ("mode" == "approx").
    The treatment of ties can be controlled by the parameter "zero_method".

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html 
    """
    input_specs = {'table' : Table}
    output_specs = {'result' : WilcoxonResult}
    config_specs = {     
        "zero_method": StrParam(default_value="wilcox", human_name="Method for zero differences treatment", short_description="Method chosen to include or not zero differences and their ranking"),
        "reference_column": StrParam(default_value="", human_name="Reference column", short_description="The reference column for pairwise comparison testing. Set empty to use the first column as reference"),
        "alternative_hypothesis": StrParam(default_value="two-sided", human_name="Alternative hypothesis", short_description="The alternative hypothesis chosen for the testing."),
        "mode": StrParam(default_value="auto", human_name="Mode", short_description="Method to calculate the p-value.")
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.to_numpy()
        data = data.T

        array_sum = np.sum(data)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            self.log_warning_message("Data contain NaN values. The paired data containing NaN values are omitted.")

        col_names = table.column_names
        ref_col = params["reference_column"]
        zero_method = params["zero_method"]
        alternat = params["alternative_hypothesis"]
        mode = params["mode"]

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
                    current_data = np.array(current_data)
                    current_data = np.transpose(current_data)
                    #------------------------
                    # removing the paired data containing NaN values from "data"
                    current_data = current_data[~np.isnan(current_data).any(1)]
                    #------------------------                
                    current_data = np.transpose(current_data)
                stat_result = wilcoxon(*current_data, zero_method=zero_method, alternative = alternat, mode=mode)
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
                    current_data = np.array(current_data)
                    current_data = np.transpose(current_data)
                    #------------------------
                    # removing the paired data containing NaN values from "data"
                    current_data = current_data[~np.isnan(current_data).any(1)]
                    #------------------------                
                    current_data = np.transpose(current_data)
                stat_result = wilcoxon(*current_data, zero_method=zero_method, alternative = alternat, mode=mode)  
                stat_result = [ref_col, col_names[i], stat_result.statistic, stat_result.pvalue]
                stat_result = np.array(stat_result)
                all_result = np.vstack((all_result, stat_result))
        
        #-------------------------------
        #suppression de la 1ère ligne artefactuelle qui a servi à initialiser "all_result"
        all_result = np.delete(all_result, 0, 0)        
        #-------------------------------
        result = WilcoxonResult(result = all_result, table=table)        
        return {'result': result}



