# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (BadRequestException, ConfigParams, FloatParam, InputSpec,
                      ListParam, OutputSpec, StrParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from statsmodels.stats.multitest import multipletests

# *****************************************************************************
#
# PValueAdujst
#
# *****************************************************************************


@task_decorator("PValueAdujst", human_name="PValue adjust",
                short_description="Test and adjust (correct) p-value for multiple tests")
class PValueAdjust(Task):
    """
    Test and adjust (correct) p-value for multiple tests

    * Input: a table containing lists of p-values.
    * Output: a table containing lists of corrected p-values.
    * Config Parameters:

      - alpha (float between 0 and 1): FWER, family-wise error rate, e.g. 0.1.
        Except for `fdr_twostage`, the p-value correction is independent of the `alpha` specified as argument
      - methods: Method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Available methods are:

        - `bonferroni`: one-step correction
        - `fdr_bh`: Benjamini/Hochberg (non-negative)
        - `fdr_by`: Benjamini/Yekutieli (negative)
        - `fdr_tsbh`: two stage fdr correction (non-negative)
        - `fdr_tsbky`: two stage fdr correction (non-negative)
        - `sidak`: one-step correction
        - `holm-sidak`: step down method using Sidak adjustments
        - `holm`: step-down method using Bonferroni adjustments
        - `simes-hochberg`: step-up method (independent)
        - `hommel`: closed method based on Simes tests (non-negative)

    For more details, see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 500

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table of p-values")}
    output_specs = {'table': OutputSpec(Table, human_name="Result",
                                        short_description="The output table containing adjected p-values")}
    config_specs = {
        "pval_column_name": StrParam(
            default_value=None, optional=True, human_name="PValue column name",
            short_description="The name of the column containing p-values. If not given, the columns with values between 0 and 1 are supposed to contain p-values."),
        "method": StrParam(
            default_value="bonferroni", human_name="Correction method",
            allowed_values=["bonferroni", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel"],
            short_description="The method used to adjust (correct) p-values"),
        "alpha": FloatParam(
            default_value=0.05, min_value=0, max_value=1, human_name="Alpha",
            short_description="FWER, family-wise error rate"),
    }

    def compute_stats(self, current_data, params: ConfigParams):
        """ compute stats """
        alpha = params.get_value("alpha")
        method = params.get_value("method")
        _, pvals_corrected, _, _ = multipletests(current_data, alpha, method)
        stat_result = pvals_corrected
        return stat_result

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        target_col_name = params.get_value("pval_column_name")
        if target_col_name:
            if target_col_name in data.columns:
                target_col_index = data.columns.get_loc(target_col_name)
                if isinstance(target_col_index, int):
                    target_col_index = [target_col_index]
            else:
                raise BadRequestException(f"The column name '{target_col_name}' does not exist")
        else:
            self.log_info_message(
                f"No column names given. The first {self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
            target_col_index = range(0, min(self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE, data.shape[1]))

        all_result = None
        for i in target_col_index:
            target_col_name = data.columns[i]
            current_data = data.iloc[:, [i]]
            tf = np.logical_and(current_data > 0, current_data < 1)
            if not tf.all(axis=None):
                self.log_warning_message(
                    f"Data of column '{target_col_name}' are not between 0 and 1. This column is omitted.")
                continue

            current_data = current_data.to_numpy().flatten()

            stat_result = self.compute_stats(current_data, params)

            if all_result is None:
                all_result = pandas.DataFrame(stat_result, columns=["Adjusted_"+target_col_name])
            else:
                df = pandas.DataFrame(stat_result, columns=["Adjusted_"+target_col_name])
                all_result = pandas.concat([all_result, df], axis=0, ignore_index=True)

        if all_result is None:
            raise BadRequestException("No valid p-value found. Please ensure that values are between 0 and 1.")

        all_result = pandas.concat([table.get_data(), all_result], axis=1)

        result_table = Table(data=all_result)
        result_table.set_all_rows_tags(table.get_row_tags())
        return {'table': result_table}
