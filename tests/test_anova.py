
import os
import numpy

from gws_core import DatasetImporter, Table
from gws_stats import KruskalWallis
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("KruskalWallis Test")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        #---------------------------------------------------------------------
        #import data
        dataset = Table.import_from_path(
            File(path=os.path.join(test_dir, "./dataset1.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0
            })
        )

        #---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params = {},
            inputs = {'dataset': dataset},
            task_type = KruskalWallis
        )
        outputs = await tester.run()
        kruskwal_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = kruskwal_result.view_scores_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
       
        print(dataset)
        print(kruskwal_result.get_result())