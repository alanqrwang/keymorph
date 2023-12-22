#!/bin/bash
#
#SBATCH --job-name=gigamed-weighted-keymorph-savebackbone # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH --exclude=ai-gpu06 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-gigamed-weighted-keymorph-savebackbone.out
#SBATCH -e ./job_err/%j-gigamed-weighted-keymorph-savebackbone.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
source activate base

#!/bin/bash

NUM_KEY=128
JOB_NAME="gigamed-weighted-keymorph"
python run.py \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --loss_fn mse \
    --kpconsistency_coeff 0 \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/ \
    --data_dir /midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed \
    --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/__training__gigamed-weighted-keymorph-savebackbone_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch350_trained_model.pth.tar \
    --train_dataset gigamed synthbrain \
    --test_dataset gigamed \
    --num_workers 4 \
    --use_amp \
    --batch_size 1 \
    --backbone conv \
    --tps_lmbda loguniform \
    --early_stop_eval_subjects 3 \
    --eval \
    --save_preds \
    --weighted_kp_align power \
    --debug_mode
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/__training__gigamed-keymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch2000_trained_model.pth.tar \