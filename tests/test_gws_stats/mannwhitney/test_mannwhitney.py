import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import MannWhitney


class TestMannWhitney(BaseTestCaseLight):

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
            params={'method': 'auto', 'alternative_hypothesis': 'two-sided'},
            inputs={'table': table},
            task_type=MannWhitney
        )
        outputs = tester.run()
        mannwhitney_result = outputs['result']
