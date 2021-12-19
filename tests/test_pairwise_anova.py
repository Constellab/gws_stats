
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import PairwiseAnova


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Pairwise ANOVA Test")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./dataset1.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={'reference_column': ""},
            inputs={'table': table},
            task_type=PairwiseAnova
        )
        outputs = await tester.run()
        anova_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=anova_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=anova_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")

        print(table)
        print(anova_result.get_result())
