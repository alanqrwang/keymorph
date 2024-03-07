#!/bin/bash

for i in 256 512 1024;
do
  sbatch /home/alw4013/keymorph/bash_scripts/train.sh "$i" 
done