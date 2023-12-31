"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""
from typing import Union
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

class Trainer(object):
    def __init__(
        self,
        epochs: int,
        device: Union[str, torch.device],
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        train_data_loader,
        val_data_loader,
        patience: int=4,
    ) -> None:
        self.epochs = epochs
        self.device = device
        self.model = model
        self.patience = patience
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.setup()

    def setup(
        self
    ) -> None:
        self.scaler = GradScaler()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params = self.model.parameters(),\
                                               weight_decay=self.weight_decay, lr=self.learning_rate)

    def fit(
       self
    ) -> None:
        best_val_loss = float('inf')
        improvement = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        for epoch in range(self.epochs):    
            epoch_loss, epoch_acc, total = self.train()
            train_loss = epoch_loss / len(self.train_data_loader)
            train_losses.append(train_loss)
            train_accs.append(epoch_acc/total)
            print(f"Epoch {epoch}/{self.epochs} train process is finished")
            print(f"Epoch {epoch}/{self.epochs},\
                                            ' train loss -> {(train_loss):.4f}")
            print(f"Epoch {epoch}/{self.epochs},\
                                ' train accuracy -> {(epoch_acc / total):.4f}")
            val_acc, val_loss = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Epoch {epoch}/{self.epochs} validation is finished")
            print(f"Epoch {epoch}/{self.epochs} validation loss -> {(val_loss):.4f}")
            print(f"Epoch {epoch}/{self.epochs}  validation accuracy -> {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                file_name = f'model-epoch={epoch}'
                file_name += f'-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.pth'
                torch.save(self.model.state_dict(), file_name)
            else:
                improvement += 1
                if improvement >= self.patience:
                    print(f'Stable loss for {self.patience} epochs.',\
                                                                'Stop training')
                    break

        self.save_train_fig(train_losses, train_accs, val_losses, val_accs)

    def train(
       self
    ) -> None:
        epoch_loss, epoch_acc, total = 0, 0, 0
        for idx, batch in tqdm(enumerate(self.train_data_loader)):
            ims, gts = batch
            ims, gts = ims.to(self.device), gts.to(self.device)
            total += ims.shape[0]
            with autocast():
                preds = self.model(ims)
                loss = self.loss(preds, gts)
                    
            self.scaler.scale(loss).backward()
                    
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            _, pred_cls = torch.max(preds.data, dim = 1)
            epoch_acc += (pred_cls == gts).sum().item()
            epoch_loss += loss.item()
        return epoch_loss, epoch_acc, total

    def validate(
        self
    ) -> None:
        self.model.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
            for idx, batch in enumerate(self.val_data_loader):
                ims, gts = batch
                ims, gts = ims.to(self.device), gts.to(self.device)
                val_total += ims.shape[0]

                with autocast():
                    preds = self.model(ims)
                    loss = self.loss(preds, gts)

                _, pred_cls = torch.max(preds.data, dim = 1)
                val_epoch_acc += (pred_cls == gts).sum().item()
                val_epoch_loss += loss.item()
            
            val_acc = val_epoch_acc / val_total
            val_loss = val_epoch_loss / len(self.val_data_loader)

            return val_acc, val_loss

    def save_train_fig(
        self,
        train_losses,
        train_accs,
        val_losses,
        val_accs
    ) -> None:
        plt.figure(figsize=(50, 40))
        plt.plot(train_losses, label='Training loss', linewidth=20)
        plt.plot(val_losses, label='Validation loss', linewidth=20)
        plt.xlabel('Epochs', fontsize=60)
        plt.ylabel('Loss', fontsize=60)
        plt.legend(fontsize=60)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        plt.savefig('train.png')

        plt.figure(figsize=(50, 40))
        # plot the training and validation accuracy
        plt.plot(train_accs, label='Training accuracy', linewidth=20)
        plt.plot(val_accs, label='Validation accuracy', linewidth=20)
        plt.xlabel('Epochs', fontsize=60)
        plt.ylabel('Accuracy', fontsize=60)
        plt.legend(fontsize=60)
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        plt.savefig("val.png")
