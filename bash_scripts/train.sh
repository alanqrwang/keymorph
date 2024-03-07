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
JOB_NAME="gigamednb-keymorph"
python run.py \
    --run_mode train \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --registration_model keymorph \
    --epochs 5000 \
    --loss_fn mse \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/ \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} \
    --train_dataset gigamednb \
    --test_dataset gigamed \
    --num_workers 4 \
    --use_amp \
    --batch_size 1 \
    --backbone conv \
    --compute_subgrids_for_tps \
    --seg_available \
    --resume_latest
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/weights/__pretrain___pretrain_gigamednb-${NUM_KEY}_datasetgigamed_modelkeymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar
    # --visualize \
    







    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining___pretrain_gigamed-synthbrain_${NUM_KEY}_datasetgigamed+synthbrain_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining__gigamed-synthbrain_datasetgigamed+synthbrain_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch8500_model.pth.tar \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/se3cnn/__pretraining__gigamed-pretraining-se3cnn_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch15000_model.pth.tar
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/__pretraining__gigamed_pretrain_slope5000_${NUM_KEY}_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/pretrained_epoch10000_model.pth.tar \