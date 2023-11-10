import os, glob
from PIL import Image
from IPython.display import display
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from data_loader import TumorDataset, DeviceDataLoader, get_default_device
from test import test
from model import ResNet18

classes = ['tumor', 'notumor']

custom_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

normalize_inverted = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

custom_ds = TumorDataset('/content/custom_dataset', custom_transform)
custom_dl = DataLoader(custom_ds, shuffle=False, num_workers=0, batch_size=1)
device = get_default_device()
custom_dl = DeviceDataLoader(custom_dl, device)

# Find the file with the maximum validation accuracy
ckpt_dir = "/content/ckpt"
files = glob.glob(os.path.join(ckpt_dir, "best_ep_*_*.pt"))
val_acc = [(float(f.split('_')[-1].replace('.pt', '')), f) for f in files] # (acc, filename) tuple
best_file = max(val_acc, key=lambda x: x[0])[1] if val_acc else None

model_path = os.path.join(ckpt_dir, best_file)
best_model = ResNet18()
best_model.load_state_dict(torch.load(model_path))

best_model.to('cuda')
test(custom_dl, best_model)

# individual check
for data in custom_dl:
    img_batch, label_batch = data
    results = best_model(img_batch)
    # confs: confidence score, preds: predicted class
    confs, preds = torch.max(results, dim=1)
    # gts: ground truth
    _, gts = torch.max(label_batch, dim=1)

    for img, p, conf, g in zip(img_batch, preds, confs, gts):
        x = normalize_inverted(img).cpu().numpy()
        x = (x*255).astype(np.uint8).T
        x = Image.fromarray(x)
        # displaying each custom image and its correctness/confidence score
        display(x)
        if p == g:
            print('True')
        else:
            print('False')
        print(f"Prediction: {classes[p]} ({conf:0.4f})")
        print(f"Ground Truth: {classes[g]}")
