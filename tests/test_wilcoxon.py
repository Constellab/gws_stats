
import os
import numpy

from gws_core import Table
from gws_stats import Wilcoxon
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Wicoxon T Test")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        #---------------------------------------------------------------------
        table = Table.import_from_path(
            File(path=os.path.join(test_dir, "./bacteria.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0
            })
        )
        #---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params = {'reference_column': "T1", 'mode': 'auto'},
            inputs = {'table': table},
            task_type = Wilcoxon
        )
        outputs = await tester.run()
        wilcoxon_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = wilcoxon_result.view_stats_result_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view = wilcoxon_result.view_stats_result_as_boxplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "box-plot-view")       
       
        print(table)
        print(wilcoxon_result.get_result())
