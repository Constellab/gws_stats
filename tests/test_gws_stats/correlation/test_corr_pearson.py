import os

from gws_core import (BaseTestCaseLight, File, Settings, TableImporter,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_stats import PearsonCorrelation


class TestPairwiseCorrelationCoef(BaseTestCaseLight):

    def test_pearson(self):
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
            params={'preselected_column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = tester.run()
        pairwise_correlationcoef_result = outputs['result']

        # ---------------------------------------------------------------------
        # run statistical test with reference_column
        tester = TaskRunner(
            params={'reference_column': 'T1', 'preselected_column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = tester.run()
        pairwise_correlationcoef_result = outputs['result']

    def test_pearson_with_group_comparison(self):
        table = DataProvider.get_iris_table()
        tester = TaskRunner(
            params={'row_tag_key': 'variety',
                    'preselected_column_names': [
                        {'name': 'petal.*', 'is_regex': True},
                        {'name': 'sepal.*', 'is_regex': True}]
                    },
            inputs={'table': table},
            task_type=PearsonCorrelation)
        outputs = tester.run()
        pairwise_correlationcoef_result = outputs['result']

        tables = pairwise_correlationcoef_result.get_group_statistics_table()
        self.assertEqual(len(tables), 3)
