#!/bin/bash
#
#SBATCH --job-name=eval # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH --exclude=ai-gpu06 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-eval.out
#SBATCH -e ./job_err/%j-eval.err \ 

module purge
module load miniconda3/22.11.1-ctkwnpe
source activate keymorph

#!/bin/bash

NUM_KEY=$1
JOB_NAME="gigamed-synthbrain-keymorph"
python run.py \
    --job_name ${JOB_NAME} \
    --num_keypoints ${NUM_KEY} \
    --max_train_keypoints 32 \
    --loss_fn mse \
    --save_dir /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/ \
    --train_dataset gigamed+synthbrain \
    --test_dataset gigamed \
    --num_workers 1 \
    --use_amp \
    --batch_size 1 \
    --backbone conv \
    --early_stop_eval_subjects 3 \
    --eval \
    --save_preds \
    --seg_available \
    --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__training__gigamed-synthbrain-keymorph_datasetgigamed+synthbrain_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch2000_trained_model.pth.tar
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/weights/__training__gigamed-keymorph_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch2000_trained_model.pth.tar \
    # --weighted_kp_align power \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/se3cnn/__training__gigamed-keymorph-se3cnn_keypoints${NUM_KEY}_batch1_normTypeinstance_lr3e-06/checkpoints/epoch2000_trained_model.pth.tar \