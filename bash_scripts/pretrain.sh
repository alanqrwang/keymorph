#!/bin/bash
#
#SBATCH --job-name=pretrain # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH --exclude=ai-gpu06 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-pretrain.out
#SBATCH -e ./job_err/%j-pretrain.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
source activate base

#!/bin/bash

NUM_KEY=$1
JOB_NAME="_pretrain_gigamed-synthbrain-randomanisotropy-${NUM_KEY}"
python /home/alw4013/keymorph/pretrain.py \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --use_wandb \
    --wandb_kwargs project=keymorph name=$JOB_NAME \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/ \
    --train_dataset gigamed+synthbrain+randomanisotropy \
    --use_amp \
    --num_workers 4 \
    --affine_slope 5000 \
    --batch_size 1 \
    --backbone conv \
    --epochs 15000 \
    --resume \
    --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining__gigamed-synthbrain-randomanisotropy_datasetgigamed+synthbrain+randomanisotropy_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch8500_model.pth.tar