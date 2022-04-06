import os

from gws_core import (BaseTestCase, ConfigParams, DatasetImporter, File, GTest,
                      Settings, Table, TableImporter, TaskRunner, ViewTester)
from gws_stats import OneWayAnova


class TestAnova(BaseTestCase):

    async def test_process(self):
        settings = Settings.retrieve()
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
            task_type=OneWayAnova
        )
        outputs = await tester.run()
        anova_result = outputs['result']

        print(table)
        print(anova_result.get_result())
