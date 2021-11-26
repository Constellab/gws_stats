
from typing import Any, List

from pandas import DataFrame
from gws_core import BoxPlotView as BoxPlotView
from gws_core import (ConfigParams)

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
