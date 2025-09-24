import sys
import torch
import argparse
import time
import ot

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import trange

sys.path.append("../lib")
from hsw import hyper_sliced_wasserstein
from hhsw import horo_hyper_sliced_wasserstein_poincare, horo_hyper_sliced_wasserstein_lorentz
from sw import sliced_wasserstein
from distributions import sampleWrappedNormal
from utils_hyperbolic import *

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


mu0 = torch.tensor([1,0,0], dtype=torch.float, device=device)
Sigma0 = torch.eye(2, dtype=torch.float, device=device)

ntry = args.ntry

ds = [3, 100]
samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)] #,int(1e6/2)]
projs = [200]

L_hsw = np.zeros((len(ds), len(projs), len(samples), ntry))
L_w = np.zeros((len(ds), len(samples), ntry))
L_s = np.zeros((len(ds), len(samples), ntry))

L_hhswp = np.zeros((len(ds), len(projs), len(samples), ntry))
L_hhswl = np.zeros((len(ds), len(projs), len(samples), ntry))
L_swl = np.zeros((len(ds), len(projs), len(samples), ntry))
L_swp = np.zeros((len(ds), len(projs), len(samples), ntry))

if __name__ == "__main__":    
    for i, d in enumerate(ds):
        for k, n_samples in enumerate(samples):
            x0 = sampleWrappedNormal(mu0, Sigma0, n_samples)
            x1 = sampleWrappedNormal(mu0, Sigma0, n_samples)
            
            if args.pbar:
                bar = trange(ntry)
            else:
                bar = range(ntry)

            for j in bar:
                for l, n_projs in enumerate(projs):
                    try:
                        t0 = time.time()
                        hsw = hyper_sliced_wasserstein(x0, x1, n_projs, device, p=2)
                        L_hsw[i, l, k, j] = time.time()-t0
                    except:
                        L_hsw[i,l,k,j] = np.inf


                    # try:
                    x0_p = lorentz_to_poincare(x0)
                    x1_p = lorentz_to_poincare(x1)
                    t0 = time.time()
                    hhsw = horo_hyper_sliced_wasserstein_poincare(x0_p, x1_p, n_projs, device, p=2)
                    L_hhswp[i,l,k,j] = time.time()-t0
                    # except:
                    #     L_hhswp[i,l,k,j] = np.inf


                    try:
                        t0 = time.time()
                        hsw = horo_hyper_sliced_wasserstein_lorentz(x0, x1, n_projs, device, p=2)
                        L_hhswl[i, l, k, j] = time.time()-t0
                    except:
                        L_hhswl[i,l,k,j] = np.inf

                    try:
                        t0 = time.time()
                        swl = sliced_wasserstein(x0, x1, n_projs, device, p=2)
                        L_swl[i, l, k, j] = time.time()-t0
                    except:
                        L_swl[i,l,k,j] = np.inf

                    try:
                        x0_p = lorentz_to_poincare(x0)
                        x1_p = lorentz_to_poincare(x1)
                        t0 = time.time()
                        swp = sliced_wasserstein(x0_p, x1_p, n_projs, device, p=2)
                        L_swp[i,l,k,j] = time.time()-t0
                    except:
                        L_swp[i,l,k,j] = np.inf

                try:
                    t2 = time.time()

                    a = torch.ones((n_samples,), device=device)/n_samples
                    b = torch.ones((n_samples,), device=device)/n_samples
                    M = torch.arccosh(torch.clamp(-minkowski_ip2(x0, x1), min=1+1e-5))**2
                    w = ot.sinkhorn2(a, b, M, reg=1, numitermax=10000, stopThr=1e-15)

                    L_s[i,k,j] = time.time()-t2
                except:
                    L_s[i,k,j] = np.inf

                try:
                    t1 = time.time()

                    a = torch.ones((n_samples,), device=device)/n_samples
                    b = torch.ones((n_samples,), device=device)/n_samples
                    M = torch.arccosh(torch.clamp(-minkowski_ip2(x0, x1), min=1+1e-5))**2
                    w = ot.emd2(a, b, M)

                    L_w[i,k,j] = time.time()-t1
                except:
                    L_w[i,k,j] = np.inf
                    
    for d in ds:
        for l, n_projs in enumerate(projs):
            np.savetxt("./Comparison_HSW_projs_"+str(n_projs)+"_d"+str(d), L_hsw[0, l])
            np.savetxt("./Comparison_HHSWp_projs_"+str(n_projs)+"_d"+str(d), L_hhswp[0, l])
            np.savetxt("./Comparison_HHSWl_projs_"+str(n_projs)+"_d"+str(d), L_hhswl[0, l])
            np.savetxt("./Comparison_SWp_projs_"+str(n_projs)+"_d"+str(d), L_swp[0, l])
            np.savetxt("./Comparison_SWl_projs_"+str(n_projs)+"_d"+str(d), L_swl[0, l])

        np.savetxt("./Comparison_SW_W_d"+str(d), L_w[0])
        np.savetxt("./Comparison_SW_Sinkhorn_d"+str(d), L_s[0])