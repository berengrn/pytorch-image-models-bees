import torch
import torch.nn as nn
import torch.nn.functional as F

from ._registry import register_model
from torchinfo import summary
from scripts import utils

class taxaNetModel(nn.Module):

    def __init__(self,hierarchy_csv):
        from timm.models import create_model
        super().__init__()  

        NbClassesLevels = utils.classCounter(hierarchy_csv)
        self.num_levels = len(NbClassesLevels)
        self.num_classes = sum(NbClassesLevels)
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,self.num_classes),
        )

    def forward(self,x):
        x = self.backbone(x)
        x =  self.classifier(x)
        return x

@register_model
def taxanet(**kwargs):
    hierarchy_csv = kwargs.pop('hierarchy')
    model = taxaNetModel(hierarchy_csv)
    return model

if __name__ == '__main__':
    from timm.models import create_model
    model = create_model('taxanet')
    summary(model)