
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import TTestOneSample


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("T Test One Sample")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./dataset5.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={'expected_value': 5},
            inputs={'dataset': table},
            task_type=TTestOneSample
        )
        outputs = await tester.run()
        ttest1samp_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=ttest1samp_result.view_scores_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        print(table)
        print(ttest1samp_result.get_result())
