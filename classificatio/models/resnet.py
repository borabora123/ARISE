import torch
import torchvision

from torch import nn
from torchvision import models


class ResNet_Model(nn.Module):
    def __init__(self,
                 model_name="resnet152",
                 n_output_layers=2,
                 output_layers_shapes=
                    {
                        5: "erosion",
                        4: "jsn"
                    },
                 pretrained=True):
        super().__init__()

        self.output_mapping = output_layers_shapes
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        elif model_name == "resnet152":
            self.backbone = models.resnet152(weights='IMAGENET1K_V2')
        else:
            raise NotImplementedError

        self.backbone.fc = nn.Sequential(
                    nn.BatchNorm1d(self.backbone.fc.in_features),
                    nn.Linear(self.backbone.fc.in_features, 1024, bias=True),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 2048, bias=True),
                    nn.BatchNorm1d(2048),
                    nn.Linear(2048, 1024, bias=True))

        self.heads = nn.ModuleList(
            [
                nn.Linear(1024, key)for key in self.output_mapping.keys()
            ]
        )
        
        self.backbone.fc.add_module('Identity', nn.Identity())
    
        
    def forward(self, x):
        feats = self.backbone(x)
        head_outputs = {
            self.output_mapping[self.heads[i].out_features]: self.heads[i](feats)
            for i in range(len(self.heads))
        }
        return head_outputs



if __name__ == "__main__":
    model = ResNet_Model()
    print(model)
    print(model(torch.randn((3, 3, 224, 224))))