import os

from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import PairwiseKruskalWallis


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("PairwiseKruskalWallis Test")
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
            params={},
            inputs={'table': table},
            task_type=PairwiseKruskalWallis
        )
        outputs = await tester.run()
        pairwise_kruskwal_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=pairwise_kruskwal_result.view_statistics_table(
                ConfigParams()
            )
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=pairwise_kruskwal_result.view_contingency_table(
                ConfigParams({
                    "metric": "p-value"
                })
            )
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        print(table)
        print(pairwise_kruskwal_result.get_result())
