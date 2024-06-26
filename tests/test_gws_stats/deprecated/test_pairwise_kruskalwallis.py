import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_stats import PairwiseKruskalWallis


class TestPairwiseKruskalWallis(BaseTestCaseLight):

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
            task_type=PairwiseKruskalWallis
        )
        outputs = tester.run()
        pairwise_kruskwal_result = outputs['result']

        print(table)
        print(pairwise_kruskwal_result.get_result())
