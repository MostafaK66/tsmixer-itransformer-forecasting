import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

import settings
from utility import Utility


class DataLoader:
    def __init__(self, name):
        self.name = name
        self.data, self.val_size, self.test_size, self.freq = self.load_data()

    def load_data(self):
        if self.name == "ettm1":
            Y_df, *_ = LongHorizon.load(directory="./", group="ETTm1")
            Y_df = Y_df[Y_df["unique_id"] == "OT"]
            Y_df["ds"] = pd.to_datetime(Y_df["ds"])
            val_size = settings.VAL_SIZE
            test_size = settings.TEST_SIZE
            freq = settings.FREQ
        elif self.name == "ettm2":
            Y_df, *_ = LongHorizon.load(directory="./", group="ETTm2")
            Y_df["ds"] = pd.to_datetime(Y_df["ds"])
            val_size = settings.VAL_SIZE
            test_size = settings.TEST_SIZE
            freq = settings.FREQ
        else:
            raise ValueError("Dataset names must be ettm1 or ettm2")
        utility = Utility()
        Y_df = utility.filter_outliers_iforest(Y_df)

        return Y_df, val_size, test_size, freq
