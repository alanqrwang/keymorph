#!/bin/bash
#
#SBATCH --job-name=gen_synthseg_brains # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-gen_synthseg_brains.out
#SBATCH -e ./job_err/%j-gen_synthseg_brains.err \ 

# if using conda
module load miniconda3/22.11.1-ctkwnpe
source activate synthseg_38
echo $PATH
# if using pip
# source ~/myvev/bin/activate

python ~/SynthSeg/scripts/tutorials/2-generation_explained.py
