"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""
from typing import Union
from tqdm import tqdm

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
        train_data_loader,
        val_data_loader,
        patience: int=4,
    ) -> None:
        self.epochs = epochs
        self.device = device
        self.model = model
        self.patience = patience
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
                                                        lr=self.learning_rate)

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
            print(f"Epoch {epoch} train process is finished")
            print(f"Epoch {epoch} train loss -> {(train_loss):.4f}")
            print(f"Epoch {epoch} train accuracy -> {(epoch_acc / total):.4f}")
            val_acc, val_loss = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                file_name = f'model-epoch={epoch}'
                file_name += f'-val_loss={val_loss}-val_acc={val_acc}.pth'
                torch.save(self.model.state_dict(), file_name)
            else:
                improvement += 1
                if improvement >= self.patience:
                    print(f'Stable loss for {self.patience} epochs.',\
                                                                'Stop training')
                    break

            print(f"Epoch {epoch} validation is finished")
            print(f"Epoch {epoch} validation loss -> {(val_loss):.4f}")
            print(f"Epoch {epoch}  validation accuracy -> {val_acc:.4f}")

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
