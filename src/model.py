"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""

import torch
import timm

class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str='resnet52',
        pretrained: bool=False
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.setup(self.model_name, self.pretrained)

    def setup(
        model_name: str,
        pretrained: bool
    ) -> None:
        self.classifier = timm.create_model(model_name=model_name,\
                                                        pretrained=pretrained)

    def forward(
        self,
        sample
    ):
        output = self.classifier(sample)
        return output

