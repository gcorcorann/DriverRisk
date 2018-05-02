#!/usr/bin/env python3
import torch
from dataloader import get_loader

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    data_path = 'data/labels_done.txt'
    batch_size = 32
    num_workers = 2

    # get dataloader
    dataloader, dataset_size = get_loaders(data_path, batch_size, num_workers,
            gpu)

if __name__ == '__main__':
    main()

