
import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import TTestOneSample


class TestTTestOneSample(BaseTestCaseLight):

    def test_process(self):
        settings = Settings.get_instance()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./dataset7.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={'expected_value': 5},
            inputs={'table': table},
            task_type=TTestOneSample
        )
        outputs = tester.run()
        ttest1samp_result = outputs['result']
