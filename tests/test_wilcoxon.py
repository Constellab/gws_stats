
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import Wilcoxon


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Wicoxon T Test")
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
            params={'mode': 'auto'},
            inputs={'table': table},
            task_type=Wilcoxon
        )
        outputs = await tester.run()
        wilcoxon_result = outputs['result']

        # ---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=wilcoxon_result.view_statistics_table(
                ConfigParams()
            )
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=wilcoxon_result.view_stats_result_as_boxplot(
                ConfigParams()
            )
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")

        print(table)
        print(wilcoxon_result.get_result())
