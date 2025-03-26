
import pandas
from gws_core import (BoolParam, ConfigParams, InputSpec, InputSpecs,
                      OutputSpec, OutputSpecs, ParamSet, StrParam, Table,
                      TableUnfolderHelper, Task, TaskInputs, TaskOutputs, ConfigSpecs,
                      resource_decorator, task_decorator)
from pandas import DataFrame
from scipy.stats import normaltest

# *****************************************************************************
#
# NormalTestResultTable
#
# *****************************************************************************


@resource_decorator("NormalTestResultTable", human_name="Normality test result", hide=True)
class NormalTestResultTable(Table):
    """ NormalTestResultTable """

# *****************************************************************************
#
# MannWhitney
#
# *****************************************************************************


@task_decorator("NormalTest", human_name="Normality test",
                short_description="Test that the distribution of a sample is normal")
class NormalTest(Task):

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 500

    """
    Test whether a sample differs from a normal distribution.

    This task tests the null hypothesis that a sample comes from a normal distribution.
    It is based on D’Agostino and Pearson’s test that combines skew and kurtosis to
    produce an omnibus test of normality.

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.
    * Config Parameters:
      - `preselected_column_names`: List of columns to pre-select for pairwise comparisons. By default a maximum pre-defined number of columns are selected (see configuration).
      - `row_tag_key`: If give, this parameter is used for group-wise testing along row tags (see example below).

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    """
    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(NormalTestResultTable, human_name="Result",
                                                     short_description="The output result")})
    config_specs = ConfigSpecs({
        "preselected_column_names":
        ParamSet(ConfigSpecs({
            "name": StrParam(
                default_value="", human_name="Pre-selected columns names", optional=True,
                short_description="The name of the column(s) to pre-select"),
            "is_regex": BoolParam(
                default_value=False, human_name="Is text pattern?",
                short_description="Set True if it is a text pattern (regular expression), False otherwise")
        }), human_name="Pre-selected column names", short_description=f"The names of column to pre-select for comparison. By default, the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used", optional=True),
        "row_tag_key":
        StrParam(
            default_value=None, optional=True, human_name="Row tag key (for group-wise testing)",
            visibility=StrParam.PROTECTED_VISIBILITY,
            short_description="The key of the row tag (representing the group axis) along which one would like to do tests")
    })

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs["table"]
        selected_cols = params.get_value("preselected_column_names")
        if selected_cols:
            table = table.select_by_column_names(selected_cols)

        row_tag_key = params.get_value("row_tag_key")
        if row_tag_key:
            result_data = self._row_group_test(table, params)
        else:
            result_data = self._column_test(table)

        result = NormalTestResultTable(data=result_data)
        return {"result": result}

    def _column_test(self, table):
        data = table.get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')
        array_has_nan = data.isnull().sum().sum()
        if array_has_nan:
            self.log_warning_message(
                "Data contain NaN values. NaN values are omitted.")

        k2, pval = normaltest(data.to_numpy(), nan_policy='omit')
        k2 = DataFrame(k2)
        pval = DataFrame(pval)
        mean = data.mean(skipna=True).to_frame()
        std = data.std(skipna=True).to_frame()
        cols = data.columns.to_frame()
        mean.index = k2.index
        std.index = k2.index
        cols.index = k2.index
        result_data = pandas.concat(
            [cols, k2, pval, mean, std], axis=1, ignore_index=True)
        result_data.columns = ["Columns",
                               "Statistics", "PValue", "Mean", "Std"]
        return result_data

    def _row_group_test(self, table, params):
        key = params.get_value("row_tag_key")
        data = table.get_data()

        all_result_data = None
        for k in range(0, data.shape[1]):
            # select each column separately to compare them
            sub_table = table.select_by_column_indexes([k])
            # unfold the current column
            sub_table = TableUnfolderHelper.unfold_rows_by_tags(
                sub_table, [key], 'column_name')
            # compare all the unfolded columns
            result_data = self._column_test(sub_table)
            if all_result_data is None:
                all_result_data = result_data
            else:
                all_result_data = pandas.concat(
                    [all_result_data, result_data], axis=0, ignore_index=True)

        return all_result_data
