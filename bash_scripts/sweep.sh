#!/bin/bash

for i in 128;
do
#   sbatch /home/alw4013/keymorph/bash_scripts/pretrain.sh "$i" 
#   sbatch /home/alw4013/keymorph/bash_scripts/train.sh "$i" 
  sbatch /home/alw4013/keymorph/bash_scripts/train_se3cnn.sh "$i" 
#   sbatch /home/alw4013/keymorph/bash_scripts/eval.sh "$i" 
done