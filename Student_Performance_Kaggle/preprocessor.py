import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Student_Performance_Kaggle.config import config
import rich
from rich.progress import track

class preprocessor:
    def __init__(self, config: 'config'):
        self.config = config
        # We pull the df from the config instance directly
        self.df = config.df.copy() 
        self.scaled_data = None
        self.scaler = StandardScaler()

    def scale_features(self, columns):
        """
        Standardizes the selected columns so they have a mean of 0 
        and a standard deviation of 1.
        """
        # with self.config.console.status('\nUsing fit transform to scale the data to similar ranges...\n', spinner='dots'):
        # 1. Extract the columns we want to cluster
        data_to_scale = self.df[columns]
        
        # 2. Fit and transform the data
        # This converts numbers like 100 and 20 into similar ranges (like 1.5 and 0.8)
        scaled_array = self.scaler.fit_transform(data_to_scale)
        
        # 3. Convert back to a DataFrame so it's easy to handle
        self.scaled_data = pd.DataFrame(scaled_array, columns=columns)
        
        return self.scaled_data