"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""
from typing import Union

import torch
import timm

__all__ = ['Model']

model_urls = {
    'vncook' : 'https://huggingface.co/tiennguyen/VNCOOK/blob/main/model-epoch%3D7-val_loss%3D0.8202-val_acc%3D0.7702.pth'
}

class Model(torch.nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        num_classes: int,
        model_name: str='resnet52',
        pretrained: bool=False
    ) -> None:
        super(Model, self).__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.setup(device, num_classes, model_name, pretrained)

    def setup(
        self,
        device: Union[str, torch.device],
        num_classes: int,
        model_name: str,
        pretrained: bool
    ) -> None:
        self.classifier = timm.create_model(model_name=model_name,\
                                            num_classes=num_classes,\
                                                        pretrained=pretrained)
        self.classifier = self.classifier.to(device)

    def forward(
        self,
        sample
    ):
        output = self.classifier(sample)
        return output

