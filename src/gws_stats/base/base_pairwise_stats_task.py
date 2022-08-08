# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod

import numpy as np
import pandas
from gws_core import (BadRequestException, BoolParam, ConfigParams, FloatParam,
                      HeatmapView, InputSpec, ListParam, OutputSpec, ParamSet,
                      StrParam, Table, TableUnfolderHelper, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from pandas import concat
from statsmodels.stats.multitest import multipletests

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult


@ task_decorator("BasePairwiseStatsTask", hide=True)
class BasePairwiseStatsTask(Task):
    """
    BasePairwiseStatsTask

    Performs pairwise comparison of the columns of a table

    # Example 1: Direct column comparisons

    Let's say you have the following table.

    | A | B | C |
    |---|---|---|
    | 1 | 5 | 3 |
    | 2 | 6 | 8 |
    | 3 | 7 | 5 |
    | 4 | 8 | 4 |

    This task performs pairwise comparison of almost all the columns of the table
    (the first 500 columns are pre-selected by default).

    - `A` will be compared with `B` and with `C`, respectively
    - `B` will be compared with `C`

    To only compare a given column with all the others, set the name of the `reference_column` (a.k.a Reference column).
    Suppose that `B` is used as reference column,only the following comaprisons will be done:

    - `B` versus `A`
    - `B` versus `C`

    It it also possible to perform comparison on a well-defined subset of the table by pre-selecting the columns of interest.
    Parameter `column_names` (a.k.a. Selected columns names) allows pre-selecting a subset of columns for analysis.

    # Example 2: Advanced comparisons along row tags

    In general, the table rows represent real-world observations (e.g. measured samples) and columns correspond to
    descriptors (a.k.a features or variables).
    Theses rows (samples) may therefore be related to metadata information given by row tags as follows:

    | row_tags                 | A | B | C |
    |--------------------------|---|---|---|
    | Gender : M <br> Age : 10 | 1 | 5 | 3 |
    | Gender : F <br> Age : 10 | 2 | 6 | 8 |
    | Gender : F <br> Age : 10 | 3 | 7 | 5 |
    | Gender : M <br> Age : 20 | 4 | 8 | 4 |

    Actually, the column ```row_tags``` does not really exist in the table. It is just to show here the tags of the rows
    Here, the first row correspond to 10-years old male individuals.

    In this this case, we may be interested in only comparing each columns along row metadata tags.
    For instance, to compare column `A` in `Males` versus `Females`, you can therefore use the advance parameters to select
    a row tag along which you would like to do such comparison.
    """

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 500
    DEFAULT_ADJUST_METHOD = "bonferroni"
    DEFAULT_ADJUST_ALPHA = 0.05

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(BasePairwiseStatsResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        "preselected_column_names":
        ParamSet({
            "name": StrParam(
                default_value="", human_name="Pre-selected columns names", optional=True,
                short_description="The name of the column(s) to pre-select"),
            "is_regex": BoolParam(
                default_value=False, human_name="Is text pattern?",
                short_description="Set True if it is a text pattern (regular expression), False otherwise")
        }, human_name="Pre-selected column names", short_description=f"The names of column to pre-select for comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used", optional=True),
        # ListParam(
        #     default_value=None, optional=True, human_name="Selected columns names",
        #     short_description=f"The names of column to pre-select for comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used"),
        "reference_column":
        StrParam(
            default_value=None, optional=True, human_name="Reference column (for column-wise comparisons)",
            short_description="The column used as reference for pairwise comparison. Only this column is compared with the others."),
        "row_tag_key":
        StrParam(
            default_value=None, optional=True, human_name="Row tag key (for group-wise comparisons)",
            visibility=StrParam.PROTECTED_VISIBILITY,
            short_description="The key of the row tag (representing the group axis) along which one would like to compare each column. This parameter is not used if a `reference column` is given."),
        "adjust_pvalue":
        ParamSet({
            "method": StrParam(
                default_value=DEFAULT_ADJUST_METHOD, human_name="Correction method",
                allowed_values=["bonferroni", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
                                "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel"],
                short_description="The method used to adjust (correct) p-values", visibility=FloatParam.PROTECTED_VISIBILITY),
            "alpha": FloatParam(
                default_value=DEFAULT_ADJUST_ALPHA, min_value=0, max_value=1, human_name="Alpha",
                short_description="FWER, family-wise error rate", visibility=StrParam.PROTECTED_VISIBILITY)
        }, human_name="Adjust p-values", short_description="Adjust p-values for multiple tests", max_number_of_occurrences=1, optional=True, visibility=ParamSet.PROTECTED_VISIBILITY)
    }

    _remove_nan_before_compute = True

    @abstractmethod
    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        """ Compute stats """
        return None

    def remove_nan(data):
        """ Remove nan """
        pass

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        reference_column = params.get_value("reference_column")
        row_tag_key = params.get_value("row_tag_key")

        if reference_column:
            all_result = self._column_wise_compare(table, params)
        elif row_tag_key:
            all_result = self._row_group_compare(table, params)
        else:
            all_result = self._column_wise_compare(table, params)

        if all_result is None:
            raise BadRequestException("The final result table seems empty. Please check pre-selected column names.")

        # adjust pvalue
        all_result = self._adjust_pvals(all_result, params)

        t = self.output_specs["result"].get_default_resource_type()
        result = t(result=all_result, input_table=table)
        return {'result': result}

    def _adjust_pvals(self, all_result, params):
        # adjust pvalue
        paraset = params.get_value("adjust_pvalue")
        if len(paraset) == 0:
            adjust_method = self.DEFAULT_ADJUST_METHOD
            adjust_alpha = self.DEFAULT_ADJUST_ALPHA
        else:
            adjust_method = paraset.get("method", self.DEFAULT_ADJUST_METHOD)
            adjust_alpha = paraset.get("alpha", self.DEFAULT_ADJUST_ALPHA)

        _, pvals_corrected, _, _ = multipletests(
            all_result.iloc[:, 3].to_numpy().flatten(),
            adjust_alpha, adjust_method)
        pvals_corrected = pandas.DataFrame(pvals_corrected)
        pvals_corrected.index = all_result.index
        all_result = pandas.concat([all_result, pvals_corrected], axis=1, ignore_index=True)

        return all_result

    def _column_wise_compare(self, table, params):

        selected_cols = params.get_value("preselected_column_names")
        reference_column = params.get_value("reference_column")
        if selected_cols:
            if reference_column:
                if reference_column in table.column_names:
                    selected_cols.append(reference_column)
            table = table.select_by_column_names(selected_cols)

        if table.nb_columns <= 1:
            raise BadRequestException(
                f"The pre-selected table contains {table.nb_columns} column(s). Please check pre-selected column name.")

        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')
        reference_column = params.get_value("reference_column")
        selected_cols = params.get_value("preselected_column_names")

        if reference_column:
            if reference_column in data.columns:
                reference_columns = [reference_column]
            else:
                raise BadRequestException(f"The reference column {reference_column} name is not found")
        else:
            reference_columns = list(set(table.column_names[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]))

        # if selected_cols:
        #     selected_cols = list(set([*selected_cols, *reference_columns]))
        # else:
        #     self.log_info_message(
        #         f"No column names given. The first {self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
        #     selected_cols = data.columns[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]
        #     selected_cols = list(set([*selected_cols, *reference_columns]))
        # data = data.loc[:, selected_cols]

        all_result = self._compare_columns(data, params, reference_columns)

        return all_result

    def _row_group_compare(self, table, params):

        selected_cols = params.get_value("preselected_column_names")
        if selected_cols:
            table = table.select_by_column_names(selected_cols)

        if table.nb_columns == 0:
            raise BadRequestException("The pre-selected table is empty. Please check pre-selected column name.")

        key = params.get_value("row_tag_key")

        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        all_result = None
        for k in range(0, data.shape[1]):
            # select each column separately to compare them
            sub_table = table.select_by_column_positions([k])

            # unfold the current column
            sub_table = TableUnfolderHelper.unfold_rows_by_tags(sub_table, [key], 'column_name')

            # compare all the unfloded columns
            reference_columns = list(set(sub_table.column_names[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]))

            if all_result is None:
                all_result = self._compare_columns(sub_table.get_data(), params, reference_columns)
            else:
                df = self._compare_columns(sub_table.get_data(), params, reference_columns)
                all_result = pandas.concat([all_result, df], axis=0, ignore_index=True)

        return all_result

    def _compare_columns(self, data, params, reference_columns=None):
        all_result = None
        is_nan_log_shown = False
        if reference_columns is None:
            reference_columns = []

        reference_column = params.get_value("reference_column")
        for i in range(0, data.shape[1]):
            ref_col = data.columns[i]
            if ref_col not in reference_columns:
                continue
            ref_data = data.iloc[:, [i]]

            for j in range(0, data.shape[1]):
                target_col = data.columns[j]
                if not reference_column:
                    if j <= i:
                        continue

                target_data = data.iloc[:, [j]]
                current_data = concat(
                    [ref_data, target_data],
                    axis=1
                )
                current_data = current_data.to_numpy().T

                array_sum = np.sum(current_data)
                array_has_nan = np.isnan(array_sum)
                if array_has_nan:
                    if self._remove_nan_before_compute:
                        current_data = [[x for x in y if not np.isnan(x)] for y in current_data]
                    if not is_nan_log_shown:
                        self.log_warning_message("Data contain NaN values. NaN values are omitted.")
                        is_nan_log_shown = True

                stat_result = self.compute_stats(current_data, ref_col, target_col, params)

                if all_result is None:
                    all_result = pandas.DataFrame([stat_result])
                else:
                    df = pandas.DataFrame([stat_result])
                    all_result = pandas.concat([all_result, df], axis=0, ignore_index=True)
        return all_result
