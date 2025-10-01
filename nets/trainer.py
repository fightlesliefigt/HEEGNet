import torch
from sklearn.metrics import balanced_accuracy_score
import yaml

from .callbacks import Callback

import traceback
import nets.functionals as fn
class Trainer:

    def __init__(self, max_epochs, callbacks, min_epochs=None, loss=None, device=None, dtype=None, swd_weight=0, lr=0.01, weight_decay=1e-3):
        self.swd_weight = swd_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_epochs = min_epochs
        self.epochs = max_epochs
        self.loss_fn = loss
        self.current_epoch = 0
        self.current_step = 0
        self.records = []
        for callback in callbacks:
            assert(isinstance(callback, Callback))
        self.callbacks = callbacks

        self.device_ = device
        self.dtype_ = dtype

        self.stop_fit_ = False
        self.optimizer = None

    def fit(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader, val_dataloader : torch.utils.data.DataLoader):

        model = model.to(dtype=self.dtype_, device=self.device_)

        self.optimizer = model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)

        [callback.on_fit_start(self, model) for callback in self.callbacks]

        for epoch in range(self.epochs):

            self.current_epoch = epoch
            [callback.on_train_epoch_start(self, model) for callback in self.callbacks]

            self.train_epoch(model, train_dataloader)

            trn_res = self.test(model, train_dataloader)
            trn_res = {f'trn_{k}': v for k, v in trn_res.items()}

            val_res = self.test(model, val_dataloader)
            val_res = {f'val_{k}': v for k, v in val_res.items()}

            self.log_dict(trn_res)
            self.log_dict(val_res)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                log_dict = trn_res | val_res
                print(f'epoch={epoch:3d} gd-step={self.current_step:5d}', end=' ')
                [print(f"{k + '=':10}{v:6.4f}", end=' ') for k,v in log_dict.items()]
                print('')                                                                 #换行


            [callback.on_train_epoch_end(self, model) for callback in self.callbacks]

            if self.stop_fit_:
                break

        [callback.on_fit_end(self, model) for callback in self.callbacks]

    def stop_fit(self):
        if self.min_epochs and self.current_epoch > self.min_epochs:
            self.stop_fit_ = True
        elif self.min_epochs is None:
            self.stop_fit_ = True
        

    def train_epoch(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader):


        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            [callback.on_train_batch_start(self, model, batch, batch_idx) for callback in self.callbacks]
            features, y = batch
            features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
            y = y.to(device=self.device_)
            #feature=features['inputs'].unsqueeze(1)  
            #domain = features['domains'].to(dtype=self.dtype_,device=self.device_)
            #features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
            y = y.to(device=self.device_)
            pred, x_flatten = model(**features)
            cn_loss = self.loss_fn(pred, y)
            if self.swd_weight !=0:
                swd_loss = fn.swd_loss(x_flatten, features['domains'])
                loss = cn_loss + self.swd_weight * swd_loss
            else:
                loss = cn_loss
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step += 1

    
    def test(self, model : torch.nn.Module, dataloader : torch.utils.data.DataLoader):

        model.eval()
        loss = 0

        y_true = []
        y_hat = []

        with torch.no_grad():
            for batch_ix, (features, y) in enumerate(dataloader):
                features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
                y = y.to(device=self.device_)
                  
                features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
                #domain = features['domains'].to(dtype=self.dtype_,device=self.device_)
                #feature=features['inputs'].unsqueeze(1)
                pred,_ = model(**features)     
                #pred = model(features['inputs'])
                loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_hat.append(pred.argmax(1))

        loss /= batch_ix + 1

        score = balanced_accuracy_score(torch.cat(y_true).detach().cpu().numpy(), torch.cat(y_hat).detach().cpu().numpy()).item()

        return dict(loss=loss, score=score)


    def log_dict(self, dictionary):
        self.records.append(dictionary | dict(epoch=self.current_epoch))
