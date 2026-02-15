#!/bin/bash

# P12
python interpretability_evaluation.py --data 'P12' --window_size 3  --batch_size 64 --hidden_size 128 \
      --feature_dim 16  --lr_cls 0.0001 --lr_rec 0.001 --lambda_cls 0.1 \
      --wd_cls 1e-4 --wd_rec 5e-4

# P19
# python interpretability_evaluation.py --data 'P19' --window_size 9  --batch_size 256 --hidden_size 512 \
#      --feature_dim 32 --lr_cls 0.00001 --lr_rec 0.001 --lambda_cls 20 \
#      --wd_cls 1e-4 --wd_rec 1e-4


