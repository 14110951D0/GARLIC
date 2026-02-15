#!/bin/bash

# P12
python main.py --data 'P12' --window_size 3  --batch_size 64 --hidden_size 128 \
      --feature_dim 16  --lr_cls 0.0001 --lr_rec 0.001 --lambda_cls 0.1 \
      --wd_cls 1e-4 --wd_rec 5e-4

# P19
# python main.py --data 'P19' --window_size 9  --batch_size 256 --hidden_size 512 \
#      --feature_dim 32 --lr_cls 0.00005 --lr_rec 0.001 --lambda_cls 10 \
#      --wd_cls 5e-4 --wd_rec 1e-4

#MIMIC-III
# python main.py --data 'MIMICIII' --window_size 5  --batch_size 128 --hidden_size 128 \
#      --feature_dim 16 --lr_cls 0.00001 --lr_rec 0.001 --lambda_cls 1 \
#      --wd_cls 1e-4 --wd_rec 1e-5

