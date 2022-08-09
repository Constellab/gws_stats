import os

from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import SpearmanCorrelation


class TestPairwiseCorrelationCoef(BaseTestCase):

    async def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./bacteria.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={'preselected_column_names': None, 'reference_column': None},
            inputs={'table': table},
            task_type=SpearmanCorrelation
        )
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']

        print(table)
        print(pairwise_correlationcoef_result.get_full_statistics_table())
        print(pairwise_correlationcoef_result.get_contingency_table(metric="pvalue"))
        print(pairwise_correlationcoef_result.get_contingency_table(metric="adjusted_pvalue"))
