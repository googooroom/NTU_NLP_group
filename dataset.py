import torch
import pandas as pd
import random
from torch.utils.data import Dataset



class TweetDisasterDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_len):
        self.annotation = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        row = self.annotation.iloc[idx]
        keyword = row['keyword']
        location = row['location']
        text = row['text']
        label = int(row['target'])


        input_text = f"{keyword if pd.notnull(keyword) else ''} {location if pd.notnull(location) else ''} {text}"

        # Tokenize the formatted_input_text
        tokens = self.tokenizer.encode_plus([input_text], add_special_tokens=True, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        label_tokens = self.tokenizer.encode_plus([label], truncation=True,  padding='max_length', max_length=self.max_len, return_tensors='pt', add_special_tokens=True)
        label_input_ids = label_tokens['input_ids']
        label_input_ids[label_input_ids == self.tokenizer.pad_token_id] = -100
        
        # Create the output dictionary
        item = {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': label_tokens['input_ids'].squeeze(),
        }
        return item
