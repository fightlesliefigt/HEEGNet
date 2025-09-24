import torch
import ot
import argparse
import sys

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from itertools import cycle
from tqdm.auto import trange
from copy import deepcopy


sys.path.append("../lib")

from utils_hyperbolic import *
from distributions import sampleWrappedNormal
from hsw import hyper_sliced_wasserstein
from sw import sliced_wasserstein
from hhsw import horo_hyper_sliced_wasserstein_lorentz, horo_hyper_sliced_wasserstein_poincare


parser = argparse.ArgumentParser()
parser.add_argument("--type_target", type=str, default="wnd", help="wnd or mwnd")
parser.add_argument("--target", type=str, default="center", help="Which target to use")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--ntry", type=int, default=5, help="number of restart")
parser.add_argument("--lr", type=float, default=1, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=10001, help="Number of epochs")
args = parser.parse_args()



device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    if args.type_target == "wnd":
        if args.target == "center":
            mu = torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64, device=device)
        elif args.target == "border":
            mu = torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64, device=device)
        Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)

    elif args.type_target == "mwnd":
        ps = np.ones(5)/5
        if args.target == "center":
            mus_lorentz = torch.tensor([[0,0.5],[0.5,0],[0,-0.5],[-0.5,0], [0,0.1]], dtype=torch.float)
        elif args.target == "border":
            mus_lorentz = torch.tensor([[0,0.9],[0.9,0],[0,-0.9],[-0.9,0], [0,0.1]], dtype=torch.float)

        mus = poincare_to_lorentz(mus_lorentz)
        sigma = 0.01 * torch.tensor([[1,0],[0,1]], dtype=torch.float)
        
        
    J = torch.diag(torch.tensor([-1,1,1], device=device, dtype=torch.float64))
    
    num_projections = 1000
    lr_hsw = args.lr
    lr_sw = args.lr
    lr_hhsw = args.lr
    n_epochs = args.n_epochs
    n_try = args.ntry
    
    
    mu0 = torch.tensor([1,0,0], dtype=torch.float64, device=device)
    Sigma0 = torch.eye(2, dtype=torch.float, device=device)
    
    L_hsw = np.zeros((n_try, n_epochs))
    L_sw = np.zeros((n_try, n_epochs))
    L_hhsw = np.zeros((n_try, n_epochs))
    L_swp = np.zeros((n_try, n_epochs)) 
    
        
    for k in range(n_try):
        if args.type_target == "wnd":
            X_target = sampleWrappedNormal(mu, Sigma, 10000)
        elif args.type_target == "mwnd":
            Z = np.random.multinomial(10000, ps)
            X = []
            for l in range(len(Z)):
                if Z[l]>0:
                    samples = sampleWrappedNormal(mus[l], sigma, int(Z[l])).numpy()
                    X += list(samples)

            X_target = torch.tensor(X, device=device, dtype=torch.float)
            
            
        train_dl = DataLoader(X_target, batch_size=500, shuffle=True)
        dataiter = iter(cycle(train_dl))
        
        x0 = sampleWrappedNormal(mu0, Sigma0, 500)
        
        x_hsw = deepcopy(x0)
        x_sw = deepcopy(x0)
        x_hhsw = deepcopy(lorentz_to_poincare(x0))
#         x_hhsw = deepcopy(x0)
        x_swp = deepcopy(lorentz_to_poincare(x0))

        x_hsw.requires_grad_(True)
        x_sw.requires_grad_(True)
        x_hhsw.requires_grad_(True)
        x_swp.requires_grad_(True)
        
        if args.pbar:
            bar = trange(n_epochs)
        else:
            bar = range(n_epochs)
            
        for e in bar:
            X_target = next(dataiter).type(torch.float64).to(device)
            
            hsw = hyper_sliced_wasserstein(x_hsw, X_target, num_projections, device, p=2)
            grad_x0_hsw = torch.autograd.grad(hsw, x_hsw)[0]
            z_hsw = torch.matmul(grad_x0_hsw, J)
            proj_hsw = z_hsw + minkowski_ip(z_hsw, x_hsw) * x_hsw
            x_hsw = expMap(-lr_hsw*proj_hsw, x_hsw)
        
            sw = sliced_wasserstein(x_sw, X_target, num_projections, device, p=2)
            grad_x0_sw = torch.autograd.grad(sw, x_sw)[0]
            z_sw = torch.matmul(grad_x0_sw, J)
            proj_sw = z_sw + minkowski_ip(z_sw, x_sw) * x_sw
            x_sw = expMap(-lr_sw*proj_sw, x_sw)
            
#             hhsw = horo_hyper_sliced_wasserstein_lorentz(x_hhsw, X_target, num_projections, device, p=2)
#             grad_x0_hhsw = torch.autograd.grad(hhsw, x_hhsw)[0]
#             z_hhsw = torch.matmul(grad_x0_hhsw, J)
#             proj_hhsw = z_hhsw + minkowski_ip(z_hhsw, x_hhsw) * x_hhsw
#             x_hhsw = expMap(-lr_hhsw*proj_hhsw, x_hhsw)
            
            hhsw = horo_hyper_sliced_wasserstein_poincare(x_hhsw, lorentz_to_poincare(X_target), 
                                                 num_projections, device, p=2)
            grad_x0_hhsw = torch.autograd.grad(hhsw, x_hhsw)[0]
            norm_x = torch.norm(x_hhsw, dim=-1, keepdim=True)
            z = (1-norm_x**2)**2/4
            x_hhsw = exp_poincare(-lr_hhsw * z * grad_x0_hhsw, x_hhsw)

            swp = sliced_wasserstein(x_swp, lorentz_to_poincare(X_target), 
                                     num_projections, device, p=2)
            grad_x0_swp = torch.autograd.grad(swp, x_swp)[0]
            norm_x = torch.norm(x_swp, dim=-1, keepdim=True)
            z = (1-norm_x**2)**2/4
            x_swp = exp_poincare(-lr_hhsw * z * grad_x0_swp, x_swp)
            
            
            n = 500           
            if args.type_target == "wnd":
                x_test = sampleWrappedNormal(mu, Sigma, n)
            elif args.type_target == "mwnd":
                Z = np.random.multinomial(n, ps)
                X = []
                for l in range(len(Z)):
                    if Z[l]>0:
                        samples = sampleWrappedNormal(mus[l], sigma, int(Z[l])).numpy()
                        X += list(samples)
                x_test = torch.tensor(X, device=device, dtype=torch.float)
                
            a = torch.ones((n,), device=device)/n
            b = torch.ones((n,), device=device)/n
            
            
            if torch.any(torch.isinf(x_sw)):
                L_sw[k, e] = np.inf
            else:
                M_sw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_sw), min=1+1e-15))**2
                w_sw = ot.emd2(a, b, M_sw)
                L_sw[k, e] = w_sw.item()
            
            if torch.any(torch.isnan(x_hsw)):
                L_hsw[k, e] = np.inf
            else:
                M_hsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_hsw), min=1+1e-15))**2
                w_hsw = ot.emd2(a, b, M_hsw)
                L_hsw[k, e] = w_hsw.item()
                
            if torch.any(torch.isnan(x_hhsw)):
                L_hhsw[k, e] = np.inf
            else:
                M_hhsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(x_hhsw)), min=1+1e-15))**2
#                 M_hhsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_hhsw), min=1+1e-15))**2
                w_hhsw = ot.emd2(a, b, M_hhsw)
                L_hhsw[k, e] = w_hhsw.item()
            
            if torch.any(torch.isnan(x_swp)):
                L_swp[k, e] = np.inf
            else:
                M_swp = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(x_swp)), min=1+1e-15))**2
#                 M_hhsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_hhsw), min=1+1e-15))**2
                w_swp = ot.emd2(a, b, M_swp)
                L_swp[k, e] = w_swp.item()
                
                
    np.savetxt("./Results/sw_loss_"+args.type_target+"_"+args.target, L_sw)
    np.savetxt("./Results/hsw_loss_"+args.type_target+"_"+args.target, L_hsw)
    np.savetxt("./Results/hhsw_loss_"+args.type_target+"_"+args.target, L_hhsw)
    np.savetxt("./Results/swp_loss_"+args.type_target+"_"+args.target, L_swp)
    
            
