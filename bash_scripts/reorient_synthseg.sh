#!/bin/bash
#
#SBATCH --job-name=reorient_synthseg # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH -p sablab-cpu # specify partition
#SBATCH --exclude ai-gpu09 # specify partition
#SBATCH -o ./job_out/%j-reorient_synthseg.out
#SBATCH -e ./job_err/%j-reorient_synthseg.err \ 

module purge
module load FSL/6.0.7-4
module load bc/1.07.1-higilk3
# if using conda
source /midtier/sablab/scratch/alw4013/miniconda3/bin/activate keymorph
# if using pip
# source ~/myvev/bin/activate

python /home/alw4013/keymorph/data_scripts/reorient_synthbrain.py
