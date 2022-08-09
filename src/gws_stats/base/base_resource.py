# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ResourceRField, ResourceSet, RField, Table,
                      resource_decorator)

@resource_decorator("BaseResource", hide=True)
class BaseResource(ResourceSet):
    _result: np.array = RField(default_value=None)
    _input_table: Table = ResourceRField()

    def __init__(self, result: np.array = None, input_table: Table = None):
        super().__init__()
        if result is not None:
            self._result = result

        if input_table is not None:
            self.input_table = input_table

    def get_result(self):
        return self._result
