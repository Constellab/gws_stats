
import numpy as np
import pandas
from gws_core import BoolParam, ParamSet, StrParam, Table
from pandas import DataFrame
from pandas.api.types import is_string_dtype

from .base_design_helper import BaseDesignHelper


class SimpleDesignHelper(BaseDesignHelper):
    CATEGROICAL_TYPE = "categorical"
    NUMERICAL_TYPE = "numerical"

    @classmethod
    def create_training_design_param_set(cls):
        return ParamSet({
            'target_name':
            StrParam(
                human_name="Target name",
                short_description="The name of the 'columns' or 'row_tag keys' to use as targets."),
            'target_origin':
            StrParam(
                default_value="column", allowed_values=["column", "row_tag"],
                human_name="Target origin",
                short_description="The origin of the target. Notice: Targets comming from a 'row_tag' are always considered as categorical."),
            'target_type':
            StrParam(
                default_value="auto", allowed_values=["auto", cls.CATEGROICAL_TYPE, cls.NUMERICAL_TYPE],
                human_name="Target type",
                short_description="The type of the target (categorical or numerical). Set 'auto' to infer the correct type. Notice: targets comming from row_tags are allways considered as categorical")},
            human_name="Training design",
            short_description="Define the training design, i.e. the target Y to use for the model.")

    @classmethod
    def create_training_matrices(cls, training_set, training_design, dummy: bool = False):
        x_true: DataFrame = training_set.get_data()
        y_tab = []
        for design_target in training_design:
            if design_target["target_origin"] == "row_tag":
                tags = training_set.get_row_tags()
                key = design_target["target_name"]
                targets = [tag[key] for tag in tags]
                # labels = sorted(list(set(targets)))
                if dummy:
                    y_temp: DataFrame = cls.convert_labels_to_dummy_matrix(
                        targets, index=training_set.row_names)
                else:
                    # y_temp: DataFrame = DataFrame(data=targets, index=training_set.row_names, columns=key)
                    y_temp = cls.convert_labels_to_numeric_matrix(
                        targets, index=training_set.row_names, columns=key)
            else:
                colname = design_target["target_name"]
                y_temp: DataFrame = training_set.select_by_column_names(
                    [{"name": colname, "is_regex": False}]).get_data()

                target_type = design_target["target_type"]

                if target_type == "auto":
                    if is_string_dtype(y_temp.to_numpy()):
                        target_type = cls.CATEGROICAL_TYPE
                    else:
                        target_type = cls.NUMERICAL_TYPE

                if target_type == cls.CATEGROICAL_TYPE:
                    y_temp = [str(k) for k in y_temp.squeeze().values]
                    if dummy:
                        y_temp = cls.convert_labels_to_dummy_matrix(
                            labels=y_temp, index=training_set.row_names)
                    else:
                        y_temp = cls.convert_labels_to_numeric_matrix(
                            labels=y_temp, index=training_set.row_names, columns=colname)

                x_true.drop(columns=[colname], inplace=True)
            y_tab.append(y_temp)

        if len(y_tab) != 0:
            y_true: DataFrame = pandas.concat(y_tab)
        else:
            y_true = None

        return x_true, y_true
