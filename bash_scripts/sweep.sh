#!/bin/bash

for i in 512 1024;
do
  sbatch /home/alw4013/keymorph/bash_scripts/eval.sh "$i" 
done