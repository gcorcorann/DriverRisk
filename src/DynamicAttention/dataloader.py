#!/usr/bin/env python3
import glob
import cv2
import numbers
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AppearanceDataset(Dataset):
    """Appearance features for risk level dataset.

    Args:
        data_path (string):     path to text file with annotations
        window_size (int):      length of sequence in window
        transform (callable):   transform to be applied to image
        display (bool):         if used for displaying (default False)

    Returns:
        torch.utils.data.Dataset:   dataset object
    """

    def __init__(self, data_path, window_size, transform=None, display=False):
        # read video paths and labels
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1, 2)

        self.data = data
        self.window_size = window_size
        self.transform = transform
        self.display = display

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # video and yolo objects path
        vid_path = self.data[idx, 0]
        obj_path = 'data/processed/objects/' + vid_path[22:-4] + '/'
        # load video
        X_frames = np.load(vid_path)
        # load labels
        y = self.data[idx, 1]
        # convert string to array of ints [0-3]
        y = np.array(list(y), dtype=int) - 1
        # grab random window
        #TODO should this be 100 - window_size + 1 ?
        if self.display == True:
            start = 0
        else:
            start = np.random.randint(100 - self.window_size)

        # window frames, objects, and labels
        X_frames = X_frames[start: start+self.window_size]
        y = y[start: start+self.window_size]
        # YOLO objects
        #TODO look into sparse matrix
        X_objs = np.zeros((self.window_size, 20, 3, 224, 224), dtype=np.float32)
        for i in range(self.window_size):
            s = obj_path + '{:02}'.format(start + i) + '-*.png'
            objs = glob.glob(s)
            if self.display == True:
                objs.sort()
            x_objs = [cv2.resize(cv2.imread(x), (224,224)) for x in objs]
            if len(x_objs) > 0 and self.transform:
                x_objs = np.array(x_objs)
                x_objs = self.transform(x_objs)
                X_objs[i][:len(x_objs)] = x_objs

        # transform data
        #TODO concatenate frames + objs then pas into transform
        if self.transform:
            X_frames = self.transform(X_frames)

        # store in sample
        sample = {'X_frames': X_frames, 'X_objs': X_objs, 'y': y}
        return sample

class CenterCrop():
    """Crop frames in video sequence at the center.
    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be center-cropped.
        
        Returns:
            ndarray: Center-cropped video.
        """
        # video dimensions
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # center-crop each frame
        video_new = video[:, top:top+new_h, left:left+new_w, :]
        return video_new

class Normalize():
    """Normalize video with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this
    transform will normalize each channge of the input video.
    Args:
        mean (list): Sequence of means for each channel.
        std (list): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be normalized
        Returns:
            ndarray: Normalized video.
        """
        video = video / 255
        video = (video - self.mean) / self.std
        video = np.asarray(video, dtype=np.float32)
        # reformat [numChannels x Height x Width]
        video = np.transpose(video, (0, 3, 1, 2))
        return video

def get_loader(data_path, sample_rate, batch_size, num_workers, display=False):
    """Return dataloader for custom dataset.

    Args:
        data_path (string):     path to data annotations
        sample_rate (int):      sample every ith frame
        batch_size (int):       number of instances in mini-batch
        num_workers (int):      number of subprocessed used for data loading
        display (bool):         if used for displaying purposes

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        int:                            dataset size
    """
    # data augmentation
    data_transforms = transforms.Compose([
        CenterCrop((224,224)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # create dataset object
    dataset = AppearanceDataset(data_path, sample_rate, data_transforms,
            display)
    dataset_size = len(dataset)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle=display,
            num_workers=num_workers)
    return dataloader, dataset_size

def main():
    """Main Function."""
    import torch
    from torchvision import utils
    import matplotlib.pyplot as plt
    
    data_path = 'data/labels_done.txt'
    window_size = 10

    batch_size = 1
    num_workers = 0

    dataloader, dataset_size = get_loader(data_path, window_size, batch_size, 
            num_workers)
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
    X_frames, X_objs, labels = batch['X_frames'], batch['X_objs'], batch['y']
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)
    print('labels:', labels.shape)
    for i in range(window_size):
        frame = X_frames[0][i]
        objs = X_objs[0][i]
        frame = frame.unsqueeze(0)
        frame = torch.cat((frame, objs))
        grid = utils.make_grid(frame)
        imshow(grid)

if __name__ == '__main__':
    main()

