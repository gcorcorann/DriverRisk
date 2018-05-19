import numpy as np
from torch.utils.data import Dataset, DataLoader

class AppearanceDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1, 2)
    
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid_path, labels = self.data[idx]
        obj_path = 'data/processed/objects/' + vid_path[22:-4] + '/'
        X_frames = np.load(vid_path)

def test():
    """Test Function."""
    data_path = 'data/labels.txt'
    dataset = AppearanceDataset(data_path)
    sample = dataset[0]

if __name__ == '__main__':
    test()

