import os, sys
import shutil
from glob import glob
import numpy as np
import random
import torch

# 재현성을 위한 랜덤시드 고정
def random_seed(random_seed):
    seed = random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Class Preprocessing; 파일 내 tumor 있는 애들 tumor로 몰아넣기
def create_symlink(img_path, from_cls, to_cls):
    src = os.path.abspath(img_path) # individual img file name
    dst = src.replace(from_cls, to_cls) # destination path
    os.makedirs(os.path.dirname(dst), exist_ok=True)  # 만약 dst
    if not os.path.exists(dst):
        os.symlink(src, dst)

def class_preprocessing():
    os.mkdir("/content/tumor")
    shutil.move("/content/Training", "/content/tumor")
    shutil.move("/content/Testing", "/content/tumor")

    dir_main = "/content/tumor"
    classes_before = ['glioma', 'meningioma', 'notumor', 'pituitary']
    classes = ['tumor', 'notumor']
    source = "Training"

    # making symbolic link to integrate as 'tumor' (찾으면 t/nt 두개 중 하나를 출력)
    for img_path in glob(os.path.join(dir_main, f"*/*/*.jpg")): #glob: 형식 맞는 파일 찾기
        for from_cls in ['glioma', 'meningioma', 'pituitary']:
            if from_cls in img_path:
                create_symlink(img_path, from_cls, 'tumor')

    image_paths = []
    for cls in classes:
        image_paths.extend(glob(os.path.join(dir_main, source, f"{cls}/*.jpg")))

'''
# 예시로 img dataset 보려면
cls_image_paths = {}
n_show = 5
for cls in classes:
    cls_image_paths[cls] = [image_path for image_path in image_paths if cls == image_path.split("/")[-2]][:n_show]

for cls in classes:
    fig, axes = plt.subplots(nrows=1, ncols=n_show, figsize=(10,2))
    for idx, image_path in enumerate(cls_image_paths[cls]):
        img = Image.open(image_path)
        axes[idx].set_title(f"{cls}_{idx}")
        axes[idx].imshow(img)
'''

# Name & Data Preprocessing
dir_main = "content/tumor"
classes = ['tumor', 'notumor']

def data_preprocessing(val_size):

    # Validation File
    os.makedirs("/content/tumor/Validating/tumor", exist_ok=True)
    os.makedirs("/content/tumor/Validating/notumor", exist_ok=True)

    val_size = val_size

    for cls in classes:
        all_images = [f for f in os.listdir(f'/content/tumor/Training/{cls}') if f.endswith('.jpg')]
        num_train = file_count_tumor = len(all_images)
        indices = list(range(num_train))
        split = int(np.floor(val_size * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        for idx in val_idx:
            if idx < len(all_images):
                source_path = os.path.join(f'/content/tumor/Training/{cls}', all_images[idx])
                dest_path = os.path.join(f'/content/tumor/Validating/{cls}', all_images[idx])
                shutil.move(source_path, dest_path)
                print(f"Moved {all_images[idx]} to Validation file")

def name_processing(dir_name, classes):
    x_data = []
    for cls in classes:
        dir_data = os.path.join(dir_main, dir_name)
        x_data.extend(glob(f"{dir_data}/{cls}/*.jpg"))
    y_data = np.array([x.split("/")[-2] for x in x_data])
    return x_data, y_data

x_train, y_train = name_processing("/content/tumor/Training", classes)
x_test, y_test = name_processing("/content/tumor/Testing", classes)
x_val, y_val = name_processing("/content/tumor/Validating", classes)

'''
# checking number of data in each folder
def get_numbers(ys, cls=None):
    cls_cnt = {}
    for y in ys:
        if y not in cls_cnt.keys():
            cls_cnt[y]=0
        cls_cnt[y]+=1
    if cls is None:
        return cls_cnt
    return cls_cnt[cls]

print(f"Class\t\tTrain\tVal\tTest\n")
for cls in classes:
    print(f"{cls:10}\t{get_numbers(y_train, cls)}\t{get_numbers(y_val, cls)}\t{get_numbers(y_test, cls)}")
'''