import os

from gws_core import BaseTestCase, File, Settings, TableImporter, TaskRunner
from gws_stats import PearsonCorrelation, PValueAdjust


class TestPValueAdjust(BaseTestCase):

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
            params={'preselected_column_names': None},
            inputs={'table': table},
            task_type=PearsonCorrelation
        )
        outputs = tester.run()
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
        outputs = tester.run()
        pairwise_correlationcoef_result = outputs['result']

        stat_table = pairwise_correlationcoef_result.get_full_statistics_table()
        print(pairwise_correlationcoef_result.get_full_statistics_table())

        # ---------------------------------------------------------------------
        # run correction test
        tester = TaskRunner(
            params={},
            inputs={'table': stat_table},
            task_type=PValueAdjust
        )
        outputs = tester.run()
        table = outputs['table']
        print(table)

        # ---------------------------------------------------------------------
        # run correction test with pval column
        tester = TaskRunner(
            params={"pval_column_name": "PValue"},
            inputs={'table': stat_table},
            task_type=PValueAdjust
        )
        outputs = tester.run()
        table = outputs['table']
        print(table)
