import torch
from torch.utils.data import Dataset, DataLoader

import re

class CustomDataset(Dataset):
    def __init__(self, review, target, tokenizer, max_len, preprocess=None):
        self.preprocess = preprocess
        self.review = review
        self.target = target.apply(lambda x: 0 if x == "negative" else 1)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        y = torch.tensor(self.target[idx], dtype=torch.float)
        X = str(self.review[idx])
        if self.preprocess:
            X = self.preprocess(X)
        
        encoded_X = self.tokenizer(
            X, 
            return_tensors = 'pt', 
            max_length = self.max_len, 
            truncation=True,
            padding = 'max_length'
            )

        return encoded_X, y



def clean(text):
    # remove weird spaces
    text =  " ".join(text.split())
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    return text