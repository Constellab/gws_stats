
import numpy as np
import pandas
from gws_core import (BoolParam, ConfigParams, FloatParam, InputSpec,
                      InputSpecs, ListParam, OutputSpec, OutputSpecs, ParamSet,
                      StrParam, Table, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
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

    ---

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.
    * Config Parameters:
      - `preselected_column_names`: List of columns to pre-select for pairwise comparisons. By default a maximum pre-defined number of columns are selected (see configuration).
      - `expected_value`: This value is compared against all the other columns means.
      - `adjust_pvalue`:
        - `method`: The correction method for p-value adjustment in multiple testing.
        - `alpha`: The FWER, family-wise error rate. Default is 0.05.
      - `alternative_hypothesis`: The alternative hypothesis chosen for the testing (`two-sided`, `less` or `greater`)

    # Example:

    Let's say you have the following table.

    | A | B | C |
    |---|---|---|
    | 1 | 5 | 3 |
    | 2 | 6 | 8 |
    | 3 | 7 | 5 |
    | 4 | 8 | 4 |

    This task performs comparisons of almost all the columns mean of the table agains an `expected_value`
    (the first `500` columns are pre-selected by default).

    The `expected_value` will be compared with the means of `A`, `B`, `C`, respectively

    ---

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    """
    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(TTestOneSampleResult, human_name="Result",
                                                     short_description="The output result")})
    config_specs = {
        "preselected_column_names":
        ParamSet({
            "name": StrParam(
                default_value="", human_name="Pre-selected columns names", optional=True,
                short_description="The name of the column(s) to pre-select"),
            "is_regex": BoolParam(
                default_value=False, human_name="Is text pattern?",
                short_description="Set True if it is a text pattern (regular expression), False otherwise")
        }, human_name="Pre-selected column names", short_description=f"The names of column to pre-select for comparison. By default, the first {BasePairwiseStatsTask.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used", optional=True),
        # "column_names": ListParam(
        #     default_value=[], human_name="Column names",
        #     short_description=f"The names of the columns to test against the expected value. By default, the first {BasePairwiseStatsTask.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used"),
        'expected_value': FloatParam(default_value=0.0, human_name="Expected value", short_description="The expected value in null hypothesis"),
        "alternative_hypothesis": StrParam(default_value="two-sided",
                                           allowed_values=[
                                               "two-sided", "less", "greater"],
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
        stat_result = ttest_1samp(
            *current_data, popmean=exp_val, alternative=alternative, nan_policy='omit')
        stat_result = [
            f"ExpectedValue = {exp_val}", target_col, stat_result.statistic, stat_result.pvalue]
        return stat_result

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        selected_cols = params.get_value("preselected_column_names")
        if selected_cols:
            table = table.select_by_column_names(selected_cols)

        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        is_nan_log_shown = False
        all_result = None

        for i in range(0, data.shape[1]):
            target_col = data.columns[i]
            current_data = data.iloc[:, [i]]
            current_data = current_data.to_numpy().T

            array_sum = np.sum(current_data)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                current_data = [
                    [x for x in y if not np.isnan(x)] for y in current_data]
                if not is_nan_log_shown:
                    self.log_warning_message(
                        "Data contain NaN values. NaN values are omitted.")
                    is_nan_log_shown = True

            stat_result = self.compute_stats(current_data, target_col, params)

            if all_result is None:
                all_result = pandas.DataFrame([stat_result])
            else:
                df = pandas.DataFrame([stat_result])
                all_result = pandas.concat(
                    [all_result, df], axis=0, ignore_index=True)

        # adjust pvalue
        all_result_dict = self._adjust_pvals(all_result, False, params)

        t = self.output_specs.get_spec("result").get_default_resource_type()
        result = t(result=all_result_dict, input_table=table)
        return {'result': result}
