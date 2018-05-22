import cv2
import glob
import numpy as np
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
        obj_path = 'data/processed/objects/' + vid_path[22:-4] + '/'
        X_frames = np.load(vid_path)
        y = np.array(list(labels), dtype=int) - 1
        # YOLO objects
        X_objs = np.zeros((100, 10, 224, 224, 3), dtype=np.float32)
        for i in range(100):
            s = obj_path + '{:02}'.format(i) + '-*.png'
            objs = glob.glob(s)
            objs.sort()
            # TODO remove if statement and just read 10 objects
            x_objs = [cv2.resize(cv2.imread(x), (224,224)) for i, x in
                    enumerate(objs) if i < 10]
            # store in array
            if len(x_objs) is not 0:
                X_objs[i][:len(x_objs)] = x_objs

        # transform data
        if self.transform:
            X_frames = self.transform(X_frames)
            #TODO don't transform all objects (since a lot of zeros)
            X_objs = self.transform(X_objs.reshape(-1, 224, 224, 3))
            X_objs = X_objs.reshape(100, 10, 3, 224, 224)

        return X_frames, X_objs, y

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

def get_loaders(train_path, valid_path, batch_size, num_workers, shuffle=True):
    data_transforms = transforms.Compose([
        CenterCrop((224,224)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    datasets = {
            'Train': AppearanceDataset(train_path, data_transforms),
            'Valid': AppearanceDataset(valid_path, data_transforms)
            }
    dataset_sizes = {x: len(datasets[x]) for x in ['Train', 'Valid']}
    dataloaders = {x: DataLoader(datasets[x], batch_size, shuffle=shuffle,
        num_workers=num_workers) for x in ['Train', 'Valid']}
    return dataloaders, dataset_sizes

def test():
    """Test Function."""
    import time

    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    batch_size = 2
    num_workers = 0
    data_loaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, num_workers)
    start = time.time()
    for data in data_loaders['Valid']:
        X_frames, X_objs, y = data
        print('X_frames:', X_frames.shape)
        print('X_objs:', X_objs.shape)
        print('y:', y.shape)
        break

    elapsed_time = time.time() - start
    print('Elapsed time:', elapsed_time)

if __name__ == '__main__':
    test()

