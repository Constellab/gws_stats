import os

from gws_core import (BaseTestCase, ConfigParams, File, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_stats import KruskalWallis


class TestKruskalWallis(BaseTestCase):

    async def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_stats:testdata_dir")
        table = TableImporter.call(
            File(path=os.path.join(test_dir, "./dataset2.csv")),
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
            task_type=KruskalWallis
        )
        outputs = await tester.run()
        kruskwal_result = outputs['result']

        print(table)
        print(kruskwal_result.get_result())
