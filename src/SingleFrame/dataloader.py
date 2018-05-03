#!/usr/bin/env python3
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    """Image Risk Level Dataset.

    Args:
        data_path (string):     path to text file with annotations
        transform (callable):   transform to be applied to image

    Returns:
        torch.utils.data.Dataset:   dataset object
    """

    def __init__(self, data_path, transform=None):
        # read video paths and labels
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1, 2)

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read video path
        vid_path = self.data[idx, 0]
        # load video
        X = np.load(vid_path)
        y = self.data[idx, 1]
        # random sample
        rand_idx = np.random.randint(len(y))
        X = X[rand_idx]
        # convert to integer ([0-3])
        y = int(y[rand_idx]) - 1
        # transform data
        if self.transform:
            X = self.transform(X)

        # store in sample
        sample = {'X': X, 'y': y}
        return sample

def get_loader(data_path, batch_size, num_workers):
    """Return dataloader for custom dataset.

    Args:
        data_path (string):     path to data annotations
        batch_size (int):       number of instances in mini-batch
        num_workers (int):      number of subprocessed used for data loading

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        int:                            dataset size
    """
    # data augmentation
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # create dataset object
    dataset = ImageDataset(data_path, data_transforms)
    dataset_size = len(dataset)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle=True,
            num_workers=num_workers)
    return dataloader, dataset_size

def main():
    """Main Function."""
    from torchvision import utils
    import matplotlib.pyplot as plt
    
    gpu = True
    data_path = 'data/labels_done.txt'
    batch_size = 32
    num_workers = 2

    dataloader, dataset_size = get_loader(data_path, batch_size, num_workers)
    print('Dataset size:', dataset_size)

    def imshow(grid):
        grid = grid.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        grid = std * grid + mean
        grid = np.clip(grid, 0, 1)
        plt.imshow(grid)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

    batch = next(iter(dataloader))
    data, labels = batch['X'], batch['y']
    print('data:', data.shape)
    print('labels:', labels.shape)
    grid = utils.make_grid(data)
    imshow(grid)

if __name__ == '__main__':
    main()

