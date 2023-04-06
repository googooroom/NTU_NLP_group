import torch
import pandas as pd
import random
from torch.utils.data import Dataset



class TweetDisasterDataset(Dataset):
    def __init__(self, csv_file: str):
        self.annotation = pd.read_csv(csv_file)
        

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        row = self.annotation.iloc[idx]
        keyword = row['keyword']
        location = row['location']
        text = row['text']
        label = row['target']
        
        input_text = f"{keyword if pd.notnull(keyword) else ''} {location if pd.notnull(location) else ''} {text}"
       
        return {
            "text_input": input_text,
            "label": label,
        }
    
    

    
    