from .batchnorm import SchedulableBatchNorm
from torch.types import Number
import torch
import os
import tempfile

class Callback:

    def on_fit_start(self, trainer, net):
        pass

    def on_train_epoch_start(self, trainer, net):
        pass

    def on_train_batch_start(self, trainer, net, batch, batch_idx):
        pass

    def on_train_epoch_end(self, trainer, net):
        pass

    def on_fit_end(self, trainer, net):
        pass

class ConstantMomentumBatchNormScheduler(Callback):
    def __init__(self, eta, eta_test) -> None:
        self.eta_ = eta
        self.eta_test_ = eta_test
        self.bn_modules_ = []


    def on_fit_start(self, trainer, net):
        if isinstance(net, torch.nn.Module):
            model = net
        else:
            raise NotImplementedError()
        # extract momentum batch norm parameters
        if model is not None:
            self.bn_modules_ = [m for m in model.modules() 
                if isinstance(m, SchedulableBatchNorm)]
        else:
            self.bn_modules_ = []

        for m in self.bn_modules_:
            m.set_eta(eta=self.eta_, eta_test = self.eta_test_)

    def __repr__(self) -> str:
        return f'ConstantMomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_test={self.eta_test_:.3f}'


class MomentumBatchNormScheduler(ConstantMomentumBatchNormScheduler):
    def __init__(self, epochs : Number, bs : Number = 32, bs0 : Number = 64, tau0 : Number = 0.9) -> None:
        assert(bs <= bs0)
        super().__init__(1. - tau0, 1. - tau0 ** (bs/bs0))
        self.epochs = epochs
        self.rho = (bs/bs0) ** (1/self.epochs)
        self.tau0 = tau0
        self.bs = bs
        self.bs0 = bs0

    def __repr__(self) -> str:
        return f'MomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_tst={self.eta_test_:.3f}'

    def on_train_epoch_start(self, trainer, net):
        self.eta_ = 1. - (self.rho ** (self.epochs * max(self.epochs - trainer.current_epoch,0)/(self.epochs-1)) - self.rho ** self.epochs)
        for m in self.bn_modules_:
            m.set_eta(eta = self.eta_)
        
        w = max(self.epochs - trainer.current_epoch,0)/(self.epochs-1)
        tau_test = self.tau0 ** (self.bs/self.bs0 * (1-w) + w * 1)
        self.eta_test_ = 1 - tau_test
        for m in self.bn_modules_:
            m.set_eta(eta_test = 1. - self.eta_test_)


class EarlyStopping(Callback):

    def __init__(self, metric='val_loss', higher_is_better=False, patience=15, verbose=False): 
                                                             
        self.tempdir = tempfile.TemporaryDirectory()     
        self.patience = patience
        self.metric = metric
        self.sign = -1 if higher_is_better else 1
        self.counter = 0
        self.best_score = self.sign * torch.Tensor([float('Inf')])
        self.best_epoch = -1
        self.verbose = verbose

    def on_train_epoch_end(self, trainer, net):                            

        current_score = self.sign * torch.Tensor([float('Inf')])
        for record in trainer.records[::-1]:                                             
            if record['epoch'] == trainer.current_epoch and self.metric in record:
                current_score = record[self.metric]
                break

        if current_score < self.best_score:      
            self.counter = 0
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
            if self.verbose:
                print(f'ES: new best score {self.best_score} for metric {self.metric} ...')
            self._save_checkpoint(net)
        else:
            self.counter += 1                 
        
        if self.counter >= self.patience:
            trainer.stop_fit()

    def _save_checkpoint(self, net):

        if self.verbose:
            print(f'ES: saving model ...')
        torch.save(net.state_dict(), os.path.join(self.tempdir.name, 'es_state_dict.pt'))  

    def on_fit_end(self, trainer, net):          
        # if early stopping was triggered
        path = os.path.join(self.tempdir.name, 'es_state_dict.pt')        
        if self.counter >= 0 and os.path.exists(path):
            if self.verbose:
                print(f'ES: loading best model ...')
            net.load_state_dict(torch.load(path))

