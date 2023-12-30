#!/bin/bash
#
#SBATCH --job-name=preprocess_gigamed # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-preprocess_gigamed.out
#SBATCH -e ./job_err/%j-preprocess_gigamed.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
module load FSL/6.0.7-4
module load bc/1.07.1-higilk3
# if using conda
source activate base
# if using pip
# source ~/myvev/bin/activate

python /home/alw4013/keymorph/data_scripts/reorient_synthbrain.py
