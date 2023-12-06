import os
import glob
from utils import images_numpy_to_tensor
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_paths = glob.glob(os.path.join(img_folder, '*.*'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = images_numpy_to_tensor(img_path).squeeze()

        if self.transform:
            img = self.transform(img)

        return img, os.path.basename(img_path)
