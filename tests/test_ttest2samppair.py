
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import TTestTwoSamplesPaired


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("T Test Two Paired Samples")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./dataset7.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={'omit_nan': True, 'reference_column': "data1"},
            inputs={'table': table},
            task_type=TTestTwoSamplesPaired
        )
        outputs = await tester.run()
        ttest2samppair_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=ttest2samppair_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=ttest2samppair_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")

        print(table)
        print(ttest2samppair_result.get_result())
