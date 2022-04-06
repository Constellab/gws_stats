
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import TTestOneSample


class TestTTestOneSample(BaseTestCase):

    async def test_process(self):
        settings = Settings.retrieve()
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
        outputs = await tester.run()
        ttest1samp_result = outputs['result']

        print(table)
        print(ttest1samp_result.get_result())
