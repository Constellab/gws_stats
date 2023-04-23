# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import BoolParam, ParamSet, StrParam, Table
from pandas import DataFrame
from pandas.api.types import is_string_dtype


class BaseDesignHelper:

    @classmethod
    def convert_labels_to_dummy_matrix(cls, labels: str, index=None) -> DataFrame:
        unique_labels = sorted(list(set(labels)))
        nb_labels = len(unique_labels)
        nb_instances = len(labels)
        data = np.zeros(shape=(nb_instances, nb_labels))
        for i in range(0, nb_instances):
            current_label = labels[i]
            idx = unique_labels.index(current_label)
            data[i][idx] = 1.0

        return DataFrame(data=data, index=index, columns=unique_labels)

    @classmethod
    def convert_labels_to_numeric_matrix(cls, labels: str, index=None, columns=None) -> DataFrame:
        unique_labels = sorted(list(set(labels)))
        nb_instances = len(labels)
        data = np.zeros(shape=(nb_instances, 1))
        for i in range(0, nb_instances):
            current_label = labels[i]
            idx = unique_labels.index(current_label)
            data[i] = idx

        return DataFrame(data=data, index=index, columns=columns)

    @classmethod
    def dummy(cls, labels: str, index=None, columns=None) -> DataFrame:
        unique_labels = sorted(list(set(labels)))
        nb_instances = len(labels)
        data = np.zeros(shape=(nb_instances, 1))
        for i in range(0, nb_instances):
            current_label = labels[i]
            idx = unique_labels.index(current_label)
            data[i] = idx

        return DataFrame(data=data, index=index, columns=columns)
