import os

from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import TTestTwoIndepSamples


class TestTTestTwoIndependantSamples(BaseTestCase):

    def test_process(self):
        settings = Settings.get_instance()
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
            params={'equal_variance': True},
            inputs={'table': table},
            task_type=TTestTwoIndepSamples
        )
        outputs = tester.run()
        ttest2sample_ind_result = outputs['result']

        print(table)
        print(ttest2sample_ind_result.get_result())
