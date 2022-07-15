# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, Table, resource_decorator, task_decorator, InputSpec, OutputSpec)
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
    ...
    * Input: a table containing the sample measurements, with the name of the samples.

    * Output: a table listing the one-way ANOVA F statistic, and the p-value for each pairwise comparison testing.

    * Config Parameters:
    - "column_names": The columns used for pairwise comparison. By default, the first three columns are used.


    For more details, see ...
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(PairwiseOneWayAnovaResult, human_name="Result", short_description="The output result")}
    config_specs = {
        **BasePairwiseStatsTask.config_specs
    }

    def compute_stats(self, current_data, ref_col, target_col, params: ConfigParams):
        stat_result = f_oneway(*current_data)
        stat_result = [ref_col, target_col, stat_result.statistic, stat_result.pvalue]
        stat_result = np.array(stat_result)
        return stat_result