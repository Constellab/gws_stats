
import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_stats import NormalTest


class TestNormalTest(BaseTestCaseLight):

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
            params={},
            inputs={'table': table},
            task_type=NormalTest
        )
        outputs = tester.run()
        normaltest_result = outputs['result']

    def test_anova_with_group_comparison(self):
        table = DataProvider.get_iris_table()
        tester = TaskRunner(
            params={'row_tag_key': 'variety',
                    'preselected_column_names': [
                        {'name': 'petal.*', 'is_regex': True},
                        {'name': 'sepal.*', 'is_regex': True}]
                    },
            inputs={'table': table},
            task_type=NormalTest)
        outputs = tester.run()
        normaltest_result = outputs['result']
