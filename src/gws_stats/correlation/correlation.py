# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, Table, resource_decorator, task_decorator, InputSpec, OutputSpec, StrParam)
from scipy.stats import (spearmanr, pearsonr)

from ..base.base_pairwise_stats_result import BasePairwiseStatsResult
from ..base.base_pairwise_stats_task import BasePairwiseStatsTask

# *****************************************************************************
#
# PairwiseCorrelationCoefResult
#
# *****************************************************************************


@resource_decorator("PairwiseCorrelationCoefResult", human_name="Result of pairwise correlation coefficient analysis", hide=True)
class PairwiseCorrelationCoefResult(BasePairwiseStatsResult):
    STATISTICS_NAME = "Statistic"

# *****************************************************************************
#
# PairwiseCorrelationCoef
#
# *****************************************************************************


@task_decorator("PairwiseCorrelationCoef", human_name="Pairwise correlation coefficient",
                short_description="Compute correlation coefficients between two groups with p-value")
class PairwiseCorrelationCoef(BasePairwiseStatsTask):
    """
    Compute the correlation coefficient (either Pearson or Spearman) for pairwise samples, from a given reference sample.

    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the correlation coefficient, and its associated p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.
    - "method": the method chosen for the computation of the correlation coefficient (either "pearson" or "spearman")

    For more details on the Pearson correlation coefficient, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html.
    For more details on the Spearman correlation coefficient, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.htm.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(PairwiseCorrelationCoefResult, human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs,
        "method": StrParam(default_value="pearson",
                        allowed_values=["pearson", "spearman"],
                        human_name="Correlation coefficient method",
                        short_description="Method chosen for the computation of the correlation coefficient"),

    }

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        method = params.get_value("method")
        if method == "pearson":
            stat_result = pearsonr(*current_data)
            stat_result = [ref_col, target_col, stat_result[0], stat_result[1]]
        else:
            stat_result = spearmanr(*current_data, nan_policy='omit')
            stat_result = [ref_col, target_col, stat_result.correlation, stat_result.pvalue]
        stat_result = np.array(stat_result)
        return stat_result