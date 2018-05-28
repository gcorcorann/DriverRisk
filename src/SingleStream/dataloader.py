#!/usr/bin/env python3
import cv2
import numbers
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AppearanceDataset(Dataset):
    """Appearance features for risk level dataset.

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
        vid_path, labels = self.data[idx]
        X_frames = np.load(vid_path)
        y = np.array(list(labels), dtype=int) - 1
        # transform data
        if self.transform:
            X_frames = self.transform(X_frames)

        return X_frames, y

class CenterCrop():
    """Crop frames in video sequence at the center.
    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, X_frames):
        """
        Args:
            video (ndarray): Video to be center-cropped.
        
        Returns:
            ndarray: Center-cropped video.
        """
        # video dimensions
        h, w = X_frames.shape[1:3]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # center-crop each frame
        X_frames = X_frames[:, top:top+new_h, left:left+new_w, :]
        return X_frames

class RandomCrop():
    """Crop randomly the frames in a video sequence.
    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, X_frames):
        """
        Args:
            video (ndarray): Video to be cropped.
        Returns:
            ndarray: Cropped video.
        """
        # video dimensions
        h, w = X_frames.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly crop each frame
        X_frames = X_frames[:, top:top+new_h, left:left+new_w, :]
        return X_frames

class RandomHorizontalFlip():
    """Horizontally flip a video sequence.
    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X_frames):
        """
        Args:
            video (ndarray): Video to be flipped.
        Returns:
            ndarray: Randomly flipped video.
        """
        # check to perform flip
        if np.random.random_sample() < self.p:
            # flip video
            video_new = np.flip(X_frames, 2)
            return X_frames

        return X_frames

class RandomRotation():
    """Rotate video sequence by an angle.
    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, X_frames):
        """
        Args:
            video (ndarray): Video to be rotated.
        Returns:
            ndarray: Randomly rotated video.
        """
        h, w = X_frames.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        # create rotation matrix with center point at the center of frame
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for i, frame in enumerate(X_frames):
            X_frames[i] = cv2.warpAffine(frame, M, (w,h))
        
        return X_frames

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

    def __call__(self, X_frames):
        """
        Args:
            video (ndarray): Video to be normalized
        Returns:
            ndarray: Normalized video.
        """
        X_frames = np.divide(X_frames, 255, dtype=np.float32)
        np.subtract(X_frames, self.mean, out=X_frames, dtype=np.float32)
        np.divide(X_frames, self.std, out=X_frames, dtype=np.float32)
        # reformat [numChannels x Height x Width]
        X_frames = np.transpose(X_frames, (0, 3, 1, 2))
        return X_frames

def get_loaders(train_path, valid_path, batch_size, num_workers, shuffle=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
            'Train': transforms.Compose([
                CenterCrop((224,224)),
                Normalize(mean, std)
                ]),
            'Valid': transforms.Compose([
                CenterCrop((224,224)),
                Normalize(mean, std)
                ])
            }
    datasets = {
            'Train': AppearanceDataset(train_path, data_transforms['Train']),
            'Valid': AppearanceDataset(valid_path, data_transforms['Valid'])
            }
    dataset_sizes = {x: len(datasets[x]) for x in ['Train', 'Valid']}
    dataloaders = {x: DataLoader(datasets[x], batch_size, shuffle=shuffle,
        num_workers=num_workers) for x in ['Train', 'Valid']}
    return dataloaders, dataset_sizes

def main():
    """Main Function."""
    from torchvision import utils
    import matplotlib.pyplot as plt
    
    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    batch_size = 1
    num_workers = 0
    data_loaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, num_workers)
    print('Dataset size:', dataset_sizes)

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

    X_frames, y = next(iter(data_loaders['Train']))
    print('X_frames:', X_frames.shape)
    print('y:', y.shape)
    grid = utils.make_grid(X_frames[0])
    imshow(grid)

if __name__ == '__main__':
    main()

