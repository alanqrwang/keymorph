#!/bin/bash

for i in 128 256;
do
  sbatch train.sh "$i" 
done