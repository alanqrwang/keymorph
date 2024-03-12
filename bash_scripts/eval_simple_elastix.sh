#!/bin/bash
#
#SBATCH --job-name=eval # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --exclude=ai-gpu06 # if you need to use a GPU
#SBATCH -p sablab-cpu # specify partition
#SBATCH -o ./job_out/%j-eval.out
#SBATCH -e ./job_err/%j-eval.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
source /midtier/sablab/scratch/alw4013/miniconda3/bin/activate keymorph

#!/bin/bash

NUM_KEY=0
JOB_NAME="itkelastix"
python run.py \
    --registration_model itkelastix \
    --run_mode eval \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --loss_fn mse \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/ \
    --train_dataset gigamed \
    --test_dataset gigamed \
    --num_workers 1 \
    --use_amp \
    --batch_size 1 \
    --backbone conv \
    --early_stop_eval_subjects 3 \
    --save_preds \
    --seg_available