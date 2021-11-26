
import os
import numpy

from gws_core import Table
from gws_stats import TTestTwoSamplesInd
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("T Test Two Independant Samples")
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
            params = {'omit_nan': True, 'reference_column': "data4", 'equal_variance': True},
            inputs = {'table': table},
            task_type = TTestTwoSamplesInd
        )
        outputs = await tester.run()
        ttest2sampind_result = outputs['result']
        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = ttest2sampind_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view = ttest2sampind_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")       

        print(table)
        print(ttest2sampind_result.get_result())
