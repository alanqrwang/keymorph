#!/bin/bash

for i in 128 256 512 1024;
do
  sbatch /home/alw4013/keymorph/slurm_scripts/pretrain.sh "$i" 
done