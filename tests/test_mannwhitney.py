
import os
import numpy

from gws_core import Table
from gws_stats import MannWhitney
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("MannWhitney U Test")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        #---------------------------------------------------------------------
        table = Table.import_from_path(
            File(path=os.path.join(test_dir, "./dataset1.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0
            })
        )

        #---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params = {'reference_column': "data1", 'method': 'auto', 'alternative_hypothesis': 'two-sided'},
            inputs = {'table': table},
            task_type = MannWhitney
        )
        outputs = await tester.run()
        mannwhitney_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = mannwhitney_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view = mannwhitney_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")

        print(table)
        print(mannwhitney_result.get_result())
