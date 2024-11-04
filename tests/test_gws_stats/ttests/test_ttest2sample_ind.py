import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import TTestTwoIndepSamples


class TestTTestTwoIndependantSamples(BaseTestCaseLight):

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
