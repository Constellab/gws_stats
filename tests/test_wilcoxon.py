
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import Wilcoxon


class TestWicoxon(BaseTestCase):

    async def test_process(self):
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
            params={'mode': 'auto'},
            inputs={'table': table},
            task_type=Wilcoxon
        )
        outputs = await tester.run()
        wilcoxon_result = outputs['result']

        print(table)
        print(wilcoxon_result.get_result())
