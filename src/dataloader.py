#!/usr/bin/env python3
import cv2
import numbers
import numpy as np
from PIL import Image
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
        # read video path
        vid_path = self.data[idx, 0]
        # load video and read label
        X = np.load(vid_path)
        y = self.data[idx, 1]
        # convert string to array of ints [0-3]
        y = np.array(list(y), dtype=int) - 1
        # transform data
        if self.transform:
            X = self.transform(X)

        # store in sample
        sample = {'X': X, 'y': y}
        return sample

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

class RandomCrop():
    """Crop randomly the frames in a video sequence.
    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be cropped.
        Returns:
            ndarray: Cropped video.
        """
        # video dimensions
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly crop each frame
        video_new = video[:, top:top+new_h, left:left+new_w, :]
        return video_new

class RandomHorizontalFlip():
    """Horizontally flip a video sequence.
    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be flipped.
        Returns:
            ndarray: Randomly flipped video.
        """
        # check to perform flip
        if np.random.random_sample() < self.p:
            # flip video
            video_new = np.flip(video, 2)
            return video_new

        return video

class RandomRotation():
    """Rotate video sequence by an angle.
    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be rotated.
        Returns:
            ndarray: Randomly rotated video.
        """
        # hold transformed video
        video_new = np.zeros_like(video)
        h, w = video.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        # create rotation matrix with center point at the center of frame
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for idx, frame in enumerate(video):
            video_new[idx] = cv2.warpAffine(frame, M, (w,h))
        
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

def get_loader(data_path, model, batch_size, num_workers, gpu):
    """Return dataloader for custom dataset.

    Args:
        data_path (string):     path to data annotations
        model (string):         model to train
        batch_size (int):       number of instances in mini-batch
        num_workers (int):      number of subprocessed used for data loading
        gpu (bool):             presence of gpu

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        int:                            dataset size
    """
    # data augmentation
    if model == 'SingleFrame':
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # create dataset object
        dataset = ImageDataset(data_path, data_transforms)
    elif model == 'SingleStream':
        data_transforms = transforms.Compose([
            CenterCrop((224,224)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # create dataset object
        dataset = AppearanceDataset(data_path, data_transforms)

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
    model = 'SingleStream'
    batch_size = 2
    num_workers = 2

    dataloader, dataset_size = get_loader(data_path, model, batch_size, 
            num_workers, gpu)
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
    grid = utils.make_grid(data[0], nrow=20)
    imshow(grid)

if __name__ == '__main__':
    main()

