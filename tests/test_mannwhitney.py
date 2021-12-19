
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import MannWhitney


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("MannWhitney U Test")
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
            params={'reference_column': "T1", 'method': 'auto', 'alternative_hypothesis': 'two-sided'},
            inputs={'table': table},
            task_type=MannWhitney
        )
        outputs = await tester.run()
        mannwhitney_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=mannwhitney_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=mannwhitney_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")

        print(table)
        print(mannwhitney_result.get_result())
