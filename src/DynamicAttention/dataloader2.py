import cv2
import glob
import numpy as np
import numbers
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AppearanceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1, 2)
    
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid_path, labels = self.data[idx]
        obj_path = vid_path[:26] + 'objects/' + vid_path[33:-4] + '/'
        X_frames = np.load(vid_path)
        y = np.array(list(labels), dtype=int) - 1
        # YOLO objects
        X_objs = np.zeros((100, 10, 256, 256, 3), dtype=np.uint8)
        for i in range(100):
            s = obj_path + '{:02}'.format(i) + '-*.png'
            objs = glob.glob(s)
            objs.sort()
            x_objs = [cv2.resize(cv2.imread(x), (256,256)) for x in objs]
            # store in array
            if len(x_objs) is not 0:
                X_objs[i][:len(x_objs)] = x_objs

        # transform data
        if self.transform:
            X_frames, X_objs = self.transform((X_frames, X_objs))

        return X_frames, X_objs, y

class RandomRotation():
    """Rotate video sequence by an angle.

    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, X):
        """
        Args:
            X (tuple): tuple of ndarrays

        Returns:
            tuple: Randomly rotated video + objects.
        """
        X_frames, X_objs = X
        # hold transformed video
        h, w = X_frames.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        # create rotation matrix with center point at the center of frame
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for i, (frame, objs) in enumerate(zip(X_frames, X_objs)):
            X_frames[i] = cv2.warpAffine(frame, M, (w,h))
            for j, obj in enumerate(objs):
                X_objs[i, j] = cv2.warpAffine(obj, M, (w, h))
        
        return X_frames, X_objs

class RandomHorizontalFlip():
    """Horizontally flip a video sequence.

    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X):
        """
        Args:
            X (tuple): tuple of ndarrays

        Returns:
            tuple: Randomly flipped video + objects.
        """
        # check to perform flip
        if np.random.random_sample() < self.p:
            X_frames, X_objs = X
            # flip video + objects
            X_frames = np.flip(X_frames, 2)
            X_objs = np.flip(X_objs, 3)
            return X_frames, X_objs

        return X

class RandomCrop():
    """Crop randomly the frames in a video sequence.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, X):
        """
        Args:
            X (tuple): tuple of ndarrays

        Returns:
            tuple: Randomly cropped video + objects.
        """
        X_frames, X_objs = X
        # crop dimensions
        h, w = X_frames.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly crop each frame
        X_frames = X_frames[:, top:top+new_h, left:left+new_w, :]
        X_objs = X_objs[:, :, top:top+new_h, left:left+new_w, :]
        return X_frames, X_objs

class CenterCrop():
    """Crop frames in video sequence at the center.
    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, X):
        """
        Args:
            X (tuple): tuple of ndarrays
        
        Returns:
            tuple: Center-cropped video + objects.
        """
        X_frames, X_objs = X
        # crop dimensions
        h, w = X_frames.shape[1:3]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # center-crop each frame
        X_frames = X_frames[:, top:top+new_h, left:left+new_w, :]
        X_objs = X_objs[:, :, top:top+new_h, left:left+new_w, :]
        return X_frames, X_objs

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

    def __call__(self, X):
        """
        Args:
            X (tuple): tuple of ndarrays

        Returns:
            tuple: Normalized video + objects.
        """
        X_frames, X_objs = X
        # normalize video
        X_frames = np.divide(X_frames, 255, dtype=np.float32)
        np.subtract(X_frames, self.mean, out=X_frames, dtype=np.float32)
        np.divide(X_frames, self.std, out=X_frames, dtype=np.float32)
        # normalize objects
        X_objs = np.divide(X_objs, 255, dtype=np.float32)
        np.subtract(X_objs, self.mean, out=X_objs, dtype=np.float32)
        np.divide(X_objs, self.std, out=X_objs, dtype=np.float32)
        # reformat [numChannels x Height x Width]
        X_frames = np.transpose(X_frames, (0, 3, 1, 2))
        X_objs = np.transpose(X_objs, (0, 1, 4, 2, 3))
        return X_frames, X_objs

def get_loaders(train_path, valid_path, batch_size, num_workers, shuffle=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
            'Train': transforms.Compose([
                RandomCrop((224,224)),
                RandomHorizontalFlip(),
                RandomRotation(15),
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

def test():
    """Test Function."""
    import matplotlib.pyplot as plt

    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    batch_size = 1
    num_workers = 0
    data_loaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, num_workers)

    def imshow(X_frames, X_objs):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        X_frames = X_frames.squeeze(0)
        X_objs = X_objs.squeeze(0)
        for frame, objs in zip(X_frames, X_objs):
            frame = frame.numpy().transpose(1,2,0)
            objs = objs.numpy().transpose(0,2,3,1)
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1) * 255
            objs = objs * std + mean
            objs = np.clip(objs, 0, 1) * 255
            for obj in objs:
                frame = np.hstack((frame, obj))

            cv2.imshow('Frame', cv2.cvtColor(np.uint8(frame),
                cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    for data in data_loaders['Train']:
        X_frames, X_objs, y = data
        print('X_frames:', X_frames.shape)
        print('X_objs:', X_objs.shape)
        print('y:', y.shape)
        imshow(X_frames, X_objs)
        break

if __name__ == '__main__':
    test()

