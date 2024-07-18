# !/bin/bash

CUDA_VISIBLE_DEVICES=$1 python demo.py --dset DOMAINNET --tid 5 --da OPDA --seed 2023 --epochs 5 --eval_epoch 1 --vlr 1e-4 --plr 1e-4 --log logs --trade 1.0
CUDA_VISIBLE_DEVICES=$1 python demo.py --dset DOMAINNET --tid 5 --da ODA  --seed 2023 --epochs 5 --eval_epoch 1 --vlr 1e-4 --plr 1e-4 --log logs --trade 1.0
CUDA_VISIBLE_DEVICES=$1 python demo.py --dset DOMAINNET --tid 5 --da CDA  --seed 2023 --epochs 5 --eval_epoch 1 --vlr 1e-4 --plr 1e-4 --log logs --trade 1.0
CUDA_VISIBLE_DEVICES=$1 python demo.py --dset DOMAINNET --tid 5 --da PDA  --seed 2023 --epochs 5 --eval_epoch 1 --vlr 1e-4 --plr 1e-4 --log logs --trade 1.0