import os

from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_stats import PearsonCorrelation


class TestPairwiseCorrelationCoef(BaseTestCase):

    async def test_pearson(self):
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
            params={'preselected_column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']

        print(table)
        print(pairwise_correlationcoef_result.get_full_statistics_table())
        print(pairwise_correlationcoef_result.get_contingency_table(metric="pvalue"))

        # ---------------------------------------------------------------------
        # run statistical test with reference_column
        tester = TaskRunner(
            params={'reference_column': 'T1', 'preselected_column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']

        print(table)
        print(pairwise_correlationcoef_result.get_full_statistics_table())
        print(pairwise_correlationcoef_result.get_contingency_table(metric="pvalue"))

    async def test_pearson_with_group_comparison(self):
        table = DataProvider.get_iris_table()
        print(table)
        tester = TaskRunner(
            params={'row_tag_key': 'variety',
                    'preselected_column_names': [
                        {'name': 'petal.*', 'is_regex': True},
                        {'name': 'sepal.*', 'is_regex': True}]
                    },
            inputs={'table': table},
            task_type=PearsonCorrelation)
        outputs = await tester.run()
        pairwise_correlationcoef_result = outputs['result']
        print(pairwise_correlationcoef_result.get_full_statistics_table())
        print(pairwise_correlationcoef_result.get_contingency_table(metric="pvalue"))

        tables = pairwise_correlationcoef_result.get_group_statistics_table()
        print(tables)
        self.assertEqual(len(tables), 3)
