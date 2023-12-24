"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""

import argparse

from torch.utils.data import DataLoader
import torchvision

from model import Model
from trainer import Trainer
from configs import Configurer

def define_trainsforms():
    transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((254,254)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.5, 0.5)
            ]
        )
    return transforms

def load_data(
    configs,
    transforms
) -> tuple:
    train = torchvision.datasets.ImageFolder(root=configs.train_dir,\
                                                        transform=transforms)
    valid = torchvision.datasets.ImageFolder(root=configs.val_dir,\
                                                        transform=transforms)
    test = torchvision.datasets.ImageFolder(root=configs.test_dir,\
                                                        transform=transforms)
    return train, valid, test

def define_data_loader(
    configs,
    train_handler,
    val_handler,
    test_handler,
):
    train_loader = DataLoader(train_handler, batch_size=configs.batch_size,\
                                        shuffle=True, num_workers=configs.cpus)
    val_loader = DataLoader(val_handler, batch_size=configs.batch_size,\
                                        shuffle=False, num_workers=configs.cpus)
    test_loader = DataLoader(test_handler, batch_size=configs.batch_size,\
                                        shuffle=False, num_workers=configs.cpus)
    return train_loader, val_loader, test_loader

def train(
    args,
    configs
) -> None:
    transforms = define_trainsforms()
    train, valid, test = load_data(configs, transforms)
    train_loader, val_loader, test_loader = define_data_loader(configs,\
                                                            train, valid, test)
    model = Model()
    trainer = Trainer(model=model, epochs=configs.epochs,\
                            patience=configs.patience, device=configs.device,\
                                        train_data_loader=train_loader,\
                                                    val_data_loader=val_loader)
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs.yaml',\
                                                    help='configuration file')
    parser.add_argument('--command', type=str, default='True',\
                                                    help='command: train, test')
    args = parser.parse_args()
    configs = Configurer(args.configs)
    eval(args.command)(args, configs)