# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any

from gws_core import BoxPlotView, ConfigParams
from pandas import DataFrame


class StatsBoxPlotView(BoxPlotView):

    _type: str = "box-plot-view"
    _stats: DataFrame = None

    def __init__(self, data: Any, stats: DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._stats = stats

    def to_dict(self, params: ConfigParams) -> dict:
        view_dict = super().to_dict(params)

        view_dict = {
            **view_dict,
            "stats": self._stats.to_dict()
        }

        return view_dict
