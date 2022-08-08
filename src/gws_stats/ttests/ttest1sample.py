# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (ConfigParams, FloatParam, InputSpec, ListParam,
                      OutputSpec, ParamSet, StrParam, Table, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from scipy.stats import ttest_1samp

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# TTestOneSampleResult
#
# *****************************************************************************


@resource_decorator("TTestOneSampleResult", human_name="T-test one sample result",
                    short_description="Result of the one-sample Student test (T-Test)", hide=True)
class TTestOneSampleResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "TStatistic"

# *****************************************************************************
#
# TtestOneSample
#
# *****************************************************************************


@task_decorator("TTestOneSample", human_name="T-test one sample",
                short_description="Test that the mean of a sample is equal to a given value")
class TTestOneSample(BasePairwiseStatsTask):
    """
    Calculate the T-test for the mean of ONE group of scores

    This is a test for the null hypothesis that the expected value (mean) of a sample of independent observations a is equal to the given population mean, popmean.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(TTestOneSampleResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        "column_names": ListParam(
            default_value=[], human_name="Column names",
            short_description=f"The names of the columns to test against the expected value. By default, the first {BasePairwiseStatsTask.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used"),
        'expected_value': FloatParam(default_value=0.0, human_name="Expected value", short_description="The expected value in null hypothesis"),
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=["two-sided", "less", "greater"],
                                           human_name="Alternative hypothesis",
                                           short_description="The alternative hypothesis chosen for the testing."),
        "adjust_pvalue":
        ParamSet({
            "method": StrParam(
                default_value=BasePairwiseStatsTask.DEFAULT_ADJUST_METHOD, human_name="Correction method",
                allowed_values=["bonferroni", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
                                "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel"],
                short_description="The method used to adjust (correct) p-values", visibility=StrParam.PROTECTED_VISIBILITY),
            "alpha": FloatParam(
                default_value=BasePairwiseStatsTask.DEFAULT_ADJUST_ALPHA, min_value=0, max_value=1, human_name="Alpha",
                short_description="FWER, family-wise error rate", visibility=FloatParam.PROTECTED_VISIBILITY)
        }, human_name="Adjust p-values", short_description="Adjust p-values for multiple tests", max_number_of_occurrences=1, optional=True, visibility=ParamSet.PROTECTED_VISIBILITY)
    }

    def compute_stats(self, current_data, target_col, params: ConfigParams):
        """ compute stats """
        exp_val = params.get_value("expected_value")
        alternative = params.get_value("alternative_hypothesis")
        stat_result = ttest_1samp(*current_data, popmean=exp_val, alternative=alternative, nan_policy='omit')
        stat_result = [f"ExpectedValue = {exp_val}", target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        target_cols = params.get_value("column_names")
        if target_cols:
            data = data.loc[:, target_cols]
        else:
            self.log_info_message(
                f"No column names given. The first {self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
            data = data.iloc[:, 0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]

        is_nan_log_shown = False
        all_result = None

        for i in range(0, data.shape[1]):
            target_col = data.columns[i]
            current_data = data.iloc[:, [i]]
            current_data = current_data.to_numpy().T

            array_sum = np.sum(current_data)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                current_data = [[x for x in y if not np.isnan(x)] for y in current_data]
                if not is_nan_log_shown:
                    self.log_warning_message("Data contain NaN values. NaN values are omitted.")
                    is_nan_log_shown = True

            stat_result = self.compute_stats(current_data, target_col, params)

            if all_result is None:
                all_result = pandas.DataFrame([stat_result])
            else:
                df = pandas.DataFrame([stat_result])
                all_result = pandas.concat([all_result, df], axis=0, ignore_index=True)

        # adjust pvalue
        all_result = self._adjust_pvals(all_result, params)

        t = self.output_specs["result"].get_default_resource_type()
        result = t(result=all_result, input_table=table)
        return {'result': result}
