
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
        table = Table.import_from_path(
            File(path=os.path.join(test_dir, "./dataset2.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0
            })
        )

        #---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params = {'omit_nan' : True},
            inputs = {'table': table},
            task_type = KruskalWallis
        )
        outputs = await tester.run()
        kruskwal_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = kruskwal_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
       
        print(table)
        print(kruskwal_result.get_result())