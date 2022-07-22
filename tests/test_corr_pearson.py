import os

from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import PearsonCorrelation


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
            params={'column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']

        print(table)
        print(pairwise_correlationcoef_result.get_statistics_table())


        # ---------------------------------------------------------------------
        # run statistical test with reference_column
        tester = TaskRunner(
            params={'reference_column': None, 'column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']

        print(table)
        print(pairwise_correlationcoef_result.get_statistics_table())
