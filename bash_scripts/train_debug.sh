#!/bin/bash
#
#SBATCH --job-name=train # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH --exclude=ai-gpu06 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-train.out
#SBATCH -e ./job_err/%j-train.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
source activate base

#!/bin/bash

NUM_KEY=128
JOB_NAME="train-debug"
python run.py \
    --job_name ${JOB_NAME} \
    --registration_model keymorph \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --loss_fn mse \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/se3cnn/ \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} \
    --train_datasets gigamed synthbrain \
    --test_dataset gigamed \
    --num_workers 4 \
    --use_amp \
    --batch_size 1 \
    --backbone se3cnn \
    --tps_lmbda loguniform \
    --compute_subgrids_for_tps \
    --visualize \
    --debug_mode
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/se3cnn/__pretraining__gigamed-pretraining-se3cnn_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar \
    # --weighted_kp_align power \

    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/__pretraining__gigamed_pretrain_slope5000_${NUM_KEY}_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/pretrained_epoch10000_model.pth.tar \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/gigamed_keymorph_256_[training]keypoints256_batch1_normTypeinstance_lr3e-06/checkpoints/epoch125_trained_model.pth.tar \

# JOB_NAME="gigamed-pretraining-se3cnn2"
# python pretraining.py \
#     --job_name ${JOB_NAME} \
#     --num_keypoints ${NUM_KEY} \
#     --use_wandb \
#     --wandb_kwargs project=keymorph name=$JOB_NAME \
#     --dataset gigamed \
#     --data_dir /midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed \
#     --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/ \
#     --use_amp \
#     --num_workers 4 \
#     --affine_slope -1 \
#     --batch_size 1 \
#     --backbone se3cnn \
#     --epochs 15000

    # --data_dir /midtier/sablab/scratch/alw4013/centered_IXI/centered_IXI/ \
