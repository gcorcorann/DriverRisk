#!/usr/bin/env python3
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    """Image Risk Level Dataset.

    Args:
        data_path (string):     path to text file with annotations
        transforms (callable):  transform to be applied on image

    Returns:
        torch.utils.data.Dataset:   dataset object
    """

    def __init__(self, data_path, transforms=None):
        # read video paths and labels
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1, 2)

        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.data[idx, 1]
        print('y:', y)

def main():
    """Main Function."""
    data_path = 'data/labels_done.txt'
    dataset = ImageDataset(data_path)
    dataset[0]

if __name__ == '__main__':
    main()

