
import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import TTestTwoRelatedSamples


class TestTTestTwoPairedSamples(BaseTestCaseLight):

    def test_process(self):
        settings = Settings.get_instance()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./bacteria.csv")),
            params={
                "delimiter": ",",
                "header": 0
            }
        )

        # ---------------------------------------------------------------------
        # run statistical test
        tester = TaskRunner(
            params={},
            inputs={'table': table},
            task_type=TTestTwoRelatedSamples
        )
        outputs = tester.run()
        ttest2sample_rel_result = outputs['result']

        print(table)
        print(ttest2sample_rel_result.get_result())
