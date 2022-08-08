# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, InputSpec, OutputSpec, StrParam, Table,
                      TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator)
from scipy.stats import spearmanr

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# SpearmanCorrelationResult
#
# *****************************************************************************


@resource_decorator("SpearmanCorrelationResult",
                    human_name="Result of pairwise spearman correlation coefficient analysis", hide=True)
class SpearmanCorrelationResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "Correlation"

# *****************************************************************************
#
# SpearmanCorrelation
#
# *****************************************************************************


@task_decorator("SpearmanCorrelation", human_name="Spearman correlation",
                short_description="Compute Spearman correlation coefficients between two groups with p-value")
class SpearmanCorrelation(BasePairwiseStatsTask):
    """
    Compute the Spearman correlation coefficient for pairwise samples, with its p-value.

    The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship
    between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are
    normally distributed. The p-value returned is a two-sided p-value.
    Like other correlation coefficients, these ones vary between -1 and +1 with 0 implying no correlation. Correlations of -1
    or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations
    imply that as x increases, y decreases.

    * Input: a table containing the sample measurements, with the name of the samples.
    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.
    * Config Parameters:

    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.

    For more details on the Spearman correlation coefficient, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.htm.
    """

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(SpearmanCorrelationResult, human_name="Result",
                                         short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
    }

    _remove_nan_before_compute = False

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        # remove nan values and clean data to have same column lengths
        idx = ~np.isnan(current_data).any(axis=0)
        current_data = current_data[:, idx]
        # compute stats
        try:
            stat_result = spearmanr(current_data[0], current_data[1])
            stat_result = [ref_col, target_col, stat_result[0], stat_result[1]]
        except Exception as _:
            stat_result = [ref_col, target_col, np.nan, np.nan]
        return stat_result
