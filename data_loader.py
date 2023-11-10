import os, sys
from typing import Tuple, List, Dict
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TumorDataset(Dataset):
    def __init__(self, dir_dataset, tr):
        self.dir_dataset = os.path.abspath(dir_dataset)
        self.classes = ['tumor', 'notumor']
        self.filelist = []
        for cls in self.classes:
            self.filelist.extend(glob(self.dir_dataset + f'/{cls}/*.jpg'))
        assert len(self.filelist) !=0, f"{self.dir_dataset + '/*/cls/*.jpg'} is empty"
        self.tr = tr

    def get_image(self, filename):
        img = Image.open(filename).convert("RGB")
        img = self.tr(img)
        return img

    def get_label(self, filename):
        label = np.array([0] * len(self.classes))
        cls = filename.split('/')[-2]
        label[self.classes.index(cls)] = 1
        return torch.from_numpy(label).type(torch.FloatTensor)


    def __getitem__(self, idx):
        filename = self.filelist[idx]
        img = self.get_image(filename)
        label = self.get_label(filename)
        return img, label

    def __len__(self):
        return len(self.filelist)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

train_transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


def get_data_loader(batch_size):
    train_ds = TumorDataset('/content/tumor/Training', train_transform)
    val_ds = TumorDataset('/content/tumor/Validating', test_transform)
    test_ds = TumorDataset('/content/tumor/Testing', test_transform)

    train_dl = DataLoader(train_ds, shuffle=True, num_workers=0, batch_size=batch_size)
    val_dl = DataLoader(val_ds, shuffle=True, num_workers=0, batch_size=batch_size)
    test_dl = DataLoader(test_ds, shuffle=True, num_workers=0, batch_size=batch_size)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    return train_dl, val_dl, test_dl
