import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter
from lib.lorentz.layers import LorentzMLR, LorentzConvGroup2d, LorentzConv2d,LorentzAvgPool2d, LorentzeLU,LorentzBatchNorm2d
from lib.lorentz.manifold import CustomLorentz
import torch.nn.functional as F
from spdnets.GyroBNH import GyroBNH2D
import spdnets.batchnorm as bn


class GyroNet(nn.Module):

    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 bnorm_dispersion= bn.BatchNormDispersion.SCALAR,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 domains = [],
                 domain_adaptation=True,
                 device='cuda',
                 dtype=torch.float64,
                 lr=0.01,
                 weight_decay=1e-3):

        super(GyroNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.bnorm_dispersion = bnorm_dispersion
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.domain_adaptation = domain_adaptation
        self.manifold = CustomLorentz()
        self.domains = domains
        self.device_ = device
        self.dtype_ = dtype
        self.lr = lr
        self.weight_decay = weight_decay
        self.manifold_violation_counters = {
            'after_projx': 0,
            'after_lc1': 0,
            'after_bn': 0,
            'after_elu': 0,
            'after_avpool': 0,
            'after_flatten': 0,
        }
        self.total_forward_calls = 0
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1,self.F1 * self.D, (self.num_electrodes, 1),
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1,bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), 
            nn.ELU(), 
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))
    

        self.ec1 =nn.Conv2d(self.F1 * self.D,self.F1 * self.D, (1, self.kernel_2),stride=1,padding=(0, self.kernel_2 // 2),bias=False,groups=self.F1 * self.D)
        self.lc1=LorentzConv2d(self.manifold, self.F1 * self.D+1, self.F2+1, 1, padding=(0, 0),bias=False, stride=1)
        if domain_adaptation:
            self.bn = bn.AdaMomDomainSPDBatchNorm(self.bn_dim(), batchdim=0, 
                                domains=self.domains,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion, 
                                eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)
        else:
            self.bn = bn.AdaMomSPDBatchNorm(self.bn_dim(), batchdim=0, 
                                          dispersion=self.bnorm_dispersion, 
                                          learn_mean=False,learn_std=True,
                                          eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)

        self.elu = LorentzeLU(self.manifold)
        self.avpool = LorentzAvgPool2d(self.manifold, (1, 8), stride=8)
        self.lmlp = LorentzMLR(self.manifold,self.feature_dim()+1, num_classes) #385,369,

    def bn_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = mock_eeg.permute(0, 2, 3, 1) 
            mock_eeg = mock_eeg / (mock_eeg.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            mock_eeg = self.manifold.projx(F.pad(mock_eeg, pad=(1, 0)))
            mock_eeg = self.lc1(mock_eeg)
            return mock_eeg.shape
        
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, inputs, domains): 
        self.total_forward_calls += 1   
        x=inputs
        x = x.unsqueeze(1)  # Add channel dimension    
        x = self.block1(x)  
        x = self.ec1(x)
        x = x.permute(0, 2, 3, 1)    
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))         
        x = self.lc1(x) 
        x = self.check_and_project_manifold(x, 'after_lc1')
        x = self.bn(x,domains)
        x = self.check_and_project_manifold(x, 'after_bn')
        x = self.elu(x)
        x = self.check_and_project_manifold(x, 'after_elu')
        x = self.avpool(x)
        x = self.check_and_project_manifold(x, 'after_avpool')
        x = self.manifold.lorentz_flatten(x)
        features = x
        x = self.check_and_project_manifold(x, 'after_flatten')
        x = self.lmlp(x)
        return x , features
    
    def check_and_project_manifold(self, x: torch.Tensor, layer_name: str):
        """
        Check if the input tensor is on the manifold and project if necessary.
        """
        ok, reason = self.manifold.check_point_on_manifold(x, explain=True)
        if not ok:
            self.manifold_violation_counters[layer_name] += 1
            #Project back to manifold
            # x = self.manifold.projx(x)
            # ok, reason = self.manifold.check_point_on_manifold(x, explain=True)
            # if not ok:
            #     raise ValueError(f"Layer {layer_name}: Input is not on the manifold after projection.")
        return x
    def print_manifold_violation_stats(self):
        """
        Print statistics about manifold violations for each layer.
        """
        print("\n" + "="*60)
        print("MANIFOLD VIOLATION STATISTICS")
        print("="*60)
        print(f"Total forward calls: {self.total_forward_calls}")
        print("-"*60)
        
        for layer_name, violation_count in self.manifold_violation_counters.items():
            violation_rate = violation_count / self.total_forward_calls * 100 if self.total_forward_calls > 0 else 0
            print(f"{layer_name:20}: {violation_count:6d} violations ({violation_rate:6.2f}%)")
        print("="*60)            


    def check_manifold(self, x: torch.Tensor):
        """
        Check if the input tensor is on the manifold.
        """
        ok, reason = self.manifold.check_point_on_manifold(x, explain=True)
        if not ok:
            print(f"Is x on manifold? {ok}, reason: {reason}")
            raise ValueError("Input is not on the manifold.")
        return ok, reason

    def configure_optimizers(self, lr=None, weight_decay=None):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('lmlp') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('lc1') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)      
            elif name.startswith('bn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)   
            elif name.startswith('avpool') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param) 
            elif name.startswith('elu') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)                                                         
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)
    
    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation:
            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)