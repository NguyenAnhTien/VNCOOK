"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""

class Trainer(object):
    def __init__(
        self,
        train_data_loader,
        val_data_loader
    ) -> None:
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def fit(
       self
    ) -> None:       
for epoch in range(epochs):    
    epoch_loss, epoch_acc, total = 0, 0, 0
    for idx, batch in tqdm(enumerate(train_dl)):
        ims, gts = batch
        ims, gts = ims.to(device), gts.to(device)
        
        total += ims.shape[0]
        
        with autocast():
            preds = resnet(ims)
            loss = loss_fn(preds, gts)
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        _, pred_cls = torch.max(preds.data, dim = 1)
        epoch_acc += (pred_cls == gts).sum().item()
        epoch_loss += loss.item()
        
    
    tr_loss = epoch_loss / len(train_dl)
    train_losses.append(tr_loss)
    train_accs.append(epoch_acc/total)
    
    print(f"Epoch {epoch + 1} train process is finished")
    print(f"Epoch {epoch + 1} train loss -> {(tr_loss):.3f}")
    print(f"Epoch {epoch + 1} train accuracy -> {(epoch_acc / total):.3f}")
    
    resnet.eval()
    with torch.no_grad():
        val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
        for idx, batch in enumerate(val_dl):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)
            val_total += ims.shape[0]

            with autocast():
                preds = resnet(ims)
                loss = loss_fn(preds, gts)

            _, pred_cls = torch.max(preds.data, dim = 1)
            val_epoch_acc += (pred_cls == gts).sum().item()
            val_epoch_loss += loss.item()
        
        val_acc = val_epoch_acc / val_total
        val_loss = val_epoch_loss / len(val_dl)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch + 1} validation is finished")
        print(f"Epoch {epoch + 1} validation loss -> {(val_loss):.3f}")
        print(f"Epoch {epoch + 1}  validation accuracy -> {val_acc:.3f}")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            improvement = 0
            torch.save(resnet.state_dict(), 'EfficientNet-B7.pth')
        else:
            improvement += 1
            if improvement >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Stopping early.")
                break


