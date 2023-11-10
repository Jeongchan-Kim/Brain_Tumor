import torch
import torch.nn as nn

'''
# Importing Model
!pip install timm
!pip install opencv-python
!pip install scikit-learn
'''

import timm

classes = ['tumor', 'notumor']

print(f"The number of pretrained models : {len(timm.list_models('*', pretrained=True))}")
timm.list_models('resnet*', pretrained=True)

model = timm.create_model('resnet18', pretrained=True)
model.default_cfg

model = timm.create_model('resnet18', pretrained=True, num_classes=len(classes), global_pool='avg')

# 랜덤으로 해당 픽셀의 이미지 넣었을 때 출력값
model.eval()
print('output shape: ', model(torch.randn(1, 3, 224, 224)).shape)

# 학습 위한 모델
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=len(classes), global_pool='avg')

    def forward(self, x):
        return torch.sigmoid(self.model(x))

model = ResNet18().to('cuda')