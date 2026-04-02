import pandas as pd
import os

class ATDataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, skiprows=2)
        rwanda_row = self.df[self.df['Countries, territories and areas'] == 'Rwanda']
        if not rwanda_row.empty:
            vision_val = rwanda_row['Training related to Vision'].values[0]
        else:
            vision_val = 'No information'
            
        self.confidence_map = {
            'Yes': 1.0,
            'Partial coverage': 0.6,
            'No': 0.2,
            'No information': 0.5
        }
        self.confidence = self.confidence_map.get(vision_val, 0.5)