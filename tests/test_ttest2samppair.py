
import os
import numpy

from gws_core import Dataset, DatasetImporter, Table
from gws_stats import TTestTwoSamplesPaired
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("T Test Two Paired Samples")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        #---------------------------------------------------------------------
        table = Table.import_from_path(
            File(path=os.path.join(test_dir, "./dataset6.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0
            })
        )
        #---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params = {},
            inputs = {'dataset': table},
            task_type = TTestTwoSamplesPaired
        )
        outputs = await tester.run()
        ttest2samppair_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = ttest2samppair_result.view_scores_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
       
        print(table)
        print(ttest2samppair_result.get_result())
