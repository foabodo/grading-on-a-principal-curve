import json
from math import ceil
import pandas as pd
from os import makedirs, path
from typing import List

class RankingDataset:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            attribute_labels: list,
            normalized_attribute_sum_name: str = "normalized_attribute_sum"
    ):
        self.attribute_labels = attribute_labels
        self.dataframe = dataframe
        self.normalized_attribute_sum_name = normalized_attribute_sum_name

        attr_max = self.dataframe[self.attribute_labels].max(axis=0)
        attr_norm = self.dataframe[self.attribute_labels].div(attr_max, axis=1)
        attr_normalized_sum = attr_norm.sum(axis=1)
        self.dataframe[self.normalized_attribute_sum_name] = attr_normalized_sum
