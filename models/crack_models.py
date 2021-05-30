
import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from swin_transformer_pytorch.swin_transformer import swin_t, swin_s, swin_b, swin_l

class CrackClassificationModels(nn.Module):
    def __init__(self,model_name,num_classes):
        super(CrackClassificationModels,self).__init__()
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc  =nn.Linear(512, 2)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc  =nn.Linear(2048, 2)
        elif model_name == "efficentnet":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        elif model_name == "swin_transformer":
            self.backbone = swin_b(num_classes=num_classes)
    def forward(self,img):
        results = self.backbone(img)
        return results