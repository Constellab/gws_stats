# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas
from gws_core import (ConfigParams, ListParam, ResourceSet, StrParam, Table,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, InputSpec, OutputSpec)
from pandas import DataFrame
from scipy.stats import normaltest

# *****************************************************************************
#
# NormalTestResultTable
#
# *****************************************************************************


@resource_decorator("NormalTestResultTable", human_name="Normality test result", hide=True)
class NormalTestResultTable(Table):
    pass

# *****************************************************************************
#
# MannWhitney
#
# *****************************************************************************


@task_decorator("NormalTest", human_name="Normality test",
                short_description="Test that the distribution of a sample is normal")
class NormalTest(Task):

    DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE = 3

    """
    Test whether a sample differs from a normal distribution.

    This task tests the null hypothesis that a sample comes from a normal distribution.
    It is based on D’Agostino and Pearson’s test that combines skew and kurtosis to
    produce an omnibus test of normality.

    For more details, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(NormalTestResultTable, human_name="Result", short_description="The output result")}
    config_specs = {
        "column_names":
        ListParam(
            default_value=None, optional=True, human_name="Columns names",
            short_description=f"The names of the columns to test. By default the first {DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE} columns are used.")
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        data = inputs["table"].get_data()
        data = data.apply(pandas.to_numeric, errors='coerce')

        column_names = params.get_value("column_names", [])
        if not column_names:
            column_names = data.columns[0:self.DEFAULT_MAX_NUMBER_OF_COLUMNS_TO_USE]
        data = data[column_names]

        k2, pval = normaltest(data.to_numpy(), nan_policy='omit')
        k2 = DataFrame(k2)
        pval = DataFrame(pval)

        mean = data.mean(skipna=True).to_frame()
        std = data.std(skipna=True).to_frame()
        mean.index = k2.index
        std.index = k2.index

        result_data = pandas.concat([k2, pval, mean, std], axis=1, ignore_index=True).T
        result_data.columns = data.columns
        result_data.index = ["Statistics", "PValue", "Mean", "Std"]

        result = NormalTestResultTable(data=result_data)
        return {"result": result}
