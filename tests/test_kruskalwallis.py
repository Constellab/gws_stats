
import os
import numpy

from gws_stats import Dataset, DatasetImporter
from gws_stats import KruskalWallis
from gws_core import Settings, GTest, BaseTestCase, TaskTester, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Test de KruskalWallis")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")

        #---------------------------------------------------------------------
        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./dataset1.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":[]
            })
        )

        # #---------------------------------------------------------------------
        # # run statistical test
        # tester = TaskTester(
        #     params = {},
        #     inputs = {'dataset': dataset},
        #     task_type = KruskalWallis
        # )
        # outputs = await tester.run()
        # trainer_result = outputs['result']

        # params = ConfigParams()
        # #---------------------------------------------------------------------
        # # test views
        # tester = ViewTester(
        #     view = trainer_result.view_scores_as_table(params)
        # )
        # dic = tester.to_dict()
        # self.assertEqual(dic["type"], "table-view")
       
        # print(trainer_result)
