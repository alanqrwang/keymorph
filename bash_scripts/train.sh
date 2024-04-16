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
source /midtier/sablab/scratch/alw4013/miniconda3/bin/activate keymorph

#!/bin/bash

NUM_KEY=$1
JOB_NAME="gigamed-noaug-${NUM_KEY}"
python run.py \
    --run_mode train \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --max_train_seg_channels 14 \
    --registration_model keymorph \
    --backbone truncatedunet \
    --num_truncated_layers_for_truncatedunet 1 \
    --epochs 5000 \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/truncatedunet1/ \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/alw4013/keymorph/wandb/ \
    --train_dataset gigamed \
    --test_dataset gigamed \
    --num_workers 4 \
    --use_amp \
    --batch_size 1 \
    --compute_subgrids_for_tps \
    --seg_available \
    --weighted_kp_align power \
    --load_path /midtier/sablab/scratch/alw4013/keymorph/weights/truncatedunet1/__pretrain___pretrain_gigamed-lesion-normal-skullstrip-nonskullstrip-${NUM_KEY}_datasetgigamed_modelkeymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/weights/truncatedunet1/__pretrain___pretrain_synthbrain-${NUM_KEY}_datasetsynthbrain_modelkeymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/truncatedunet1/__training__gigamed-weighted-lesion-normal-skullstrip-nonskullstrip-${NUM_KEY}_datasetgigamed_modelkeymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch5000_trained_model.pth.tar
    # --resume_latest
    # --visualize \
    







    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining___pretrain_gigamed-synthbrain_${NUM_KEY}_datasetgigamed+synthbrain_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining__gigamed-synthbrain_datasetgigamed+synthbrain_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch8500_model.pth.tar \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/se3cnn/__pretraining__gigamed-pretraining-se3cnn_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/__pretraining__gigamed_pretrain_slope5000_${NUM_KEY}_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/pretrained_epoch10000_model.pth.tar \