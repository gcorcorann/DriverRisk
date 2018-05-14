#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class data(Dataset):
    def __init__(self, inputs, labels):
        self.x = inputs
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = [1, 3, 3]
        return X
        

def main():
    x = [1, 2, 3]
    y = [1, 2, 3]
    d = data(x, y)
    d1 = DataLoader(d, batch_size=2)
    for item in d1:
        print(item[0].shape)
        break
        

if __name__ == '__main__':
    main()
