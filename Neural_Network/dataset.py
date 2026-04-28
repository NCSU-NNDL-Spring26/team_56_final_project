import torch
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    def __init__(self, X, y):
        
        #X: numpy array of shape (N, signal_length)
        #y: numpy array of shape (N,)
       
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
