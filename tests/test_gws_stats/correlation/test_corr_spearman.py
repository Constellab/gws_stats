import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import SpearmanCorrelation


class TestPairwiseCorrelationCoef(BaseTestCaseLight):

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
            params={'preselected_column_names': None, 'reference_column': None},
            inputs={'table': table},
            task_type=SpearmanCorrelation
        )
        outputs = tester.run()
        pairwise_correlationcoef_result = outputs['result']
