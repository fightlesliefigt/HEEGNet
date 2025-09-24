#!/bin/bash

## CIFAR10
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 2 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 3 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 4 --lambd 1 --scale_var 0.1 --prop 0.75

python train_busemann.py --dataset "cifar10" --loss "hhsw_mixt" --dims 2 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hhsw_mixt" --dims 3 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hhsw_mixt" --dims 4 --lambd 1 --scale_var 0.1 --prop 0.75


## CIFAR100
python train_busemann.py --dataset "cifar100" --loss "hsw_mixt" --dims 3 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "hsw_mixt" --dims 5 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "hsw_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75


python train_busemann.py --dataset "cifar100" --loss "hhsw_mixt" --dims 3 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "hhsw_mixt" --dims 5 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "hhsw_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75

