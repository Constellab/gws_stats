import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_stats import OneWayAnova


class TestAnova(BaseTestCaseLight):

    def test_anova(self):
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
            task_type=OneWayAnova
        )
        outputs = tester.run()
        anova_result = outputs['result']

        print(table)
        print(anova_result.get_statistics_table())

    def test_anova_with_group_comparison(self):
        table = DataProvider.get_iris_table()
        tester = TaskRunner(
            params={'row_tag_key': 'variety',
                    'preselected_column_names': [
                        {'name': 'petal.*', 'is_regex': True},
                        {'name': 'sepal.*', 'is_regex': True}]
                    },
            inputs={'table': table},
            task_type=OneWayAnova)
        outputs = tester.run()
        result = outputs['result']

        print(table)
        print(result.get_statistics_table())
