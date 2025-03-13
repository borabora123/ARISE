import torch
import torchvision

from torch import nn
from torchvision import models


class Conv_Next(nn.Module):
    def __init__(self,
                 model_name="convnext_small",
                 n_output_layers=2,
                 output_layers_shapes=
                    {
                        5: "erosion",
                        4: "jsn"
                    },
                 pretrained=True):
        super().__init__()

        self.output_mapping = output_layers_shapes
        self.backbone = models.convnext_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(768),
                    nn.Linear(768, 768, bias=True),
                    nn.Dropout(0.25),
                    nn.Linear(768, 1024, bias=True),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512, bias=True))
        self.heads = nn.ModuleList(
            [
                nn.Linear(512, key)for key in self.output_mapping.keys()
            ]
        )
        
        self.backbone.classifier.add_module('Identity', nn.Identity())
    
        
    def forward(self, x):
        feats = self.backbone(x)
        head_outputs = {
            self.output_mapping[self.heads[i].out_features]: self.heads[i](feats)
            for i in range(len(self.heads))
        }
        return head_outputs



if __name__ == "__main__":
    model = Conv_Next()
    print(model)
    print(model(torch.randn((3, 3, 224, 224))))