import torch
from torch.utils.data.dataloader import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, idx):
        item = {key : torch.tensor(value[idx]) for key, value in self.tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
