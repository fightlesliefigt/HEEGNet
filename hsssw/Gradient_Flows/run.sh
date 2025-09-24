#!/bin/bash

## For WNDs:
python xp_gradientflows.py --type_target "wnd" --target "center" --ntry 5 --lr 5 --n_epochs 5001
python xp_gradientflows.py --type_target "wnd" --target "border" --ntry 5 --lr 5 --n_epochs 5001

## For mixture of WNDs
python xp_gradientflows.py --type_target "mwnd" --target "center" --ntry 5 --lr 1 --n_epochs 10001
python xp_gradientflows.py --type_target "mwnd" --target "border" --ntry 5 --lr 1 --n_epochs 10001
