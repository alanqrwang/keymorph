# KeyMorph: Robust and Flexible Multi-modal Registration via Keypoint Detection

KeyMorph is a deep learning-based image registration framework that relies on automatically extracting corresponding keypoints. 
It supports unimodal/multimodal pairwise and groupwise registration using rigid, affine, or nonlinear transformations.
This repository contains the code for KeyMorph, as well as example scripts for training your own KeyMorph model.
<!-- As an example, it uses data from the [IXI dataset](https://brain-development.org/ixi-dataset/) to train and evaluate the model. -->

### BrainMorph
[BrainMorph](https://github.com/alanqrwang/brainmorph) is a foundation model based on the KeyMorph framework, trained on over 100,000 brain MR images at full resolution (256x256x256).
The model is robust to normal and diseased brains, a variety of MRI modalities, and skullstripped and non-skullstripped images.
Check out the dedicated repository for the latest updates and models!

## Updates
- [June 2024] Added support for keypoint alignment in real-world space. Use this if you are registering between different volumes in the same series, and there is a common reference space that the volumes share. See "Training KeyMorph on your own data" below for more details.
- [May 2024] Added option to use CSV file to train KeyMorph on your own data. See "Training KeyMorph on your own data" below.
- [May 2024] [BrainMorph](https://github.com/alanqrwang/brainmorph) has been moved to its own dedicated repository. See the repository for the latest updates and models.
- [May 2024] [BrainMorph](https://github.com/alanqrwang/brainmorph) is released, a foundational keypoint model based on KeyMorph for robust and flexible brain MRI registration!
- [Dec 2023] [Journal paper](https://arxiv.org/abs/2304.09941) extension of MIDL paper published in Medical Image Analysis. Instructions under "IXI-trained, half-resolution models".
- [Feb 2022] [Conference paper](https://openreview.net/forum?id=OrNzjERFybh) published in MIDL 2021.

## Installation
To run scripts and/or contribute to keymorph, you should install from source:
```bash
git clone https://github.com/alanqrwang/keymorph.git
cd keymorph
pip install -e .
```

You can also install keymorph using using pip:
```bash
pip install keymorph
```


### Requirements
The keymorph package depends on the following requirements:

- numpy>=1.19.1
- ogb>=1.2.6
- outdated>=0.2.0
- pandas>=1.1.0
- pytz>=2020.4
- torch>=1.7.0
- torchvision>=0.8.2
- scikit-learn>=0.20.0
- scipy>=1.5.4
- torchio>=0.19.6

Running `pip install keymorph` or `pip install -e .` will automatically check for and install all of these requirements.


## TLDR in code
The crux of the code is in the `forward()` function in `keymorph/model.py`, which performs one forward pass through the entire KeyMorph pipeline.

Here's a pseudo-code version of the function:
```python
def forward(img_f, img_m, seg_f, seg_m, network, optimizer, kp_aligner):
    '''Forward pass for one mini-batch step. 
    Variables with (_f, _m, _a) denotes (fixed, moving, aligned).
    
    Args:
        img_f, img_m: Fixed and moving intensity image (bs, 1, l, w, h)
        seg_f, seg_m: Fixed and moving one-hot segmentation map (bs, num_classes, l, w, h)
        network: Keypoint extractor network
        kp_aligner: Rigid, affine or TPS keypoint alignment module
    '''
    optimizer.zero_grad()

    # Extract keypoints
    points_f = network(img_f)
    points_m = network(img_m)

    # Align via keypoints
    grid = kp_aligner.grid_from_points(points_m, points_f, img_f.shape, lmbda=lmbda)
    img_a, seg_a = utils.align_moving_img(grid, img_m, seg_m)

    # Compute losses
    mse = MSELoss()(img_f, img_a)
    soft_dice = DiceLoss()(seg_a, seg_f)

    if unsupervised:
        loss = mse
    else:
        loss = soft_dice

    # Backward pass
    loss.backward()
    optimizer.step()
```
The `network` variable is a CNN with center-of-mass layer which extracts keypoints from the input images.
The `kp_aligner` variable is a keypoint alignment module. It has a function `grid_from_points()` which returns a flow-field grid encoding the transformation to perform on the moving image. The transformation can either be rigid, affine, or nonlinear (TPS).

## Training KeyMorph on your own data
`scripts/run.py` with `--run_mode train` allows you to easily train KeyMorph on your own data.

### Create a CSV file for your data
#### Option 1: Explicit pairs
The simplest way is to create a CSV with the following columns: 
+ `fixed_img_path` 
+ `moving_img_path` 
+ `fixed_seg_path` 
+ `moving_seg_path` 
+ `fixed_mask_path` 
+ `moving_mask_path` 
+ `train`

For each row, `fixed_*_path` and `moving_*_path` are paths to the pair of fixed/moving images, segmentations, and masks, respectively.
That is, every fixed/moving pair must be an explicit row in the CSV.
If segmentations or masks are not available, set those entries to "None".
`train` is a boolean indicating whether the image pair is in the train or test set.

Then, simply pass the path to the CSV file as `--data_csv_path`.

#### Option 2. Modality-based
Alternatively, instead of explicitly enumerating all pairs you want to train, you can just include all img/seg/mask paths in your dataset and the code can automatically sample pairs for you.
A `modality` column can enable additional control of how pairs are sampled (see Other optional flags below).
The CSV file should contain the following columns: `img_path`, `seg_path`, `mask_path`, `modality`, `train`.
+ `img_path` is the path to the intensity image.
+ `seg_path` is the (optional) path to the corresponding segmentation map. Set to "None" if not available.
+ `mask_path` is the (optional) path to the mask. Set to "None" if not available.
+ `modality` is the modality of the image.
+ `train` is a boolean indicating whether the image is in the train or test set.

Then, simply pass the path to the CSV file as `--data_csv_path`.

### Editing pre-processing/augmentations
The `scripts/hyperparameters.py` file contains all hyperparameters for training KeyMorph.
For training, the most important you'll need to set is the `TRANSFORM` variable, which corresponds to the pre-processing/augmentations that you want to apply to your own data.
The code uses `torchio` for pre-processing and augmentations. You can find the list of available transforms [here](https://torchio.readthedocs.io/transforms/augmentation.html).
You can also use your own custom transforms by wrapping them in the [Lambda](https://torchio.readthedocs.io/transforms/others.html#torchio.transforms.Lambda) transform.

Note, affine augmentations are applied separately and is determined by the `--max_random_affine_augment_params` flag in `scripts/run.py`. By default, it is set to `(0.0, 0.0, 0.0, 0.0)`. For example, `(0.2, 0.2, 3.1416, 0.1)` denotes:
+ Scaling by `[1-0.2, 1+0.2]`
+ Translation by `[-0.2, 0.2]`, as a fraction of the image size
+ Rotation by `[-3.1416, 3.1416]` radians
+ Shearing by `[-0.1, 0.1]`

### Run the training script
```bash
python scripts/run.py \
    --run_mode train \
    --num_keypoints 128 \
    --loss_fn mse \
    --transform_type affine \
    --backbone truncatedunet \
    --use_amp \
    --train_dataset csv \
    --data_path /path/to/data_csv 
```


Description of all flags:
+ `--num_keypoints <num_key>` flag specifies the number of keypoints to extract per image as `<num_key>`.
+ `--loss_fn <loss>` specifies the loss function to train. Options are `mse` (unsupervised training) and `dice` (supervised training). Unsupervised only requires intensity images and minimizes MSE loss, while supervised assumes availability of corresponding segmentation maps for each image and minimizes soft Dice loss.
+ `--transform_type <ttype>`. 
Transform to use for registration. Options are `rigid`, `affine`, `tps_<lambda>`.
TPS uses a (non-linear) thin-plate-spline interpolant to align the corresponding keypoints. A hyperparameter lambda controls the degree of non-linearity for TPS. A value of 0 corresponds to exact keypoint alignment (resulting in a maximally nonlinear transformation while still minimizing bending energy), while higher values result in the transformation becoming more and more affine-like. In practice, we find a value of 10 is very similar to an affine transformation.
The code also supports sampling lambda according to some distribution (`tps_uniform`, `tps_lognormal`, `tps_loguniform`).
+ `--backbone truncatedunet` sets the keypoint extractor backbone to a truncated U-Net. Other options are `conv` (a simple convolutional encoder network) and `unet` (a full U-Net).
+ `--use_amp` flag to use automatic mixed precision training.
+ `--train_dataset csv` specifies that we are training on a csv dataset specified by...
+ `--data_path <path>` specifies the path to the CSV file containing the dataset.

Other optional flags:
+  `--mix_modalities` flag, if set, mixes modalities between sampled pairs during training. You should probably set this when `--loss_fn dice` (supervised training), and not when `--loss_fn mse` (unsupervised training).
+ `--visualize` flag to visualize results with matplotlib
+ `--debug_mode` flag to print some debugging information
+ `--use_wandb` flag to log results to Weights & Biases

#### Keypoint alignment in real-world space
In aforementioned cases, keypoints are extracted and aligned in a normalized `[-1, 1]` space.
In cases where you are registering multiple views of the same subject in a series, you can align keypoints in *real-world space*.
This reference space is commonly recorded for medical images and is encoded by an affine matrix associated with each volume, which maps from *voxel coordinates/indices* to real world space (which usually has units in millimeters).

This code has support for aligning keypoints in real-world space. To do this, simply add the flag `--align_keypoints_in_real_world_space`. Currently, only rigid and affine transformations are supported for real-world space alignment, but TPS is coming soon!

Note, this flag should not be used if:
1. you are registering between different subjects
2. you don't have the affine matrices (or you suspect they are inaccurate) for each volume

## Downloading Trained Weights
You can find all full-resolution, BrainMorph trained weights [here](https://cornell.box.com/s/2mw4ey1u7waqrpylnxf49rck7u3nnr7i).
Half-resolution trained weights are under [Releases](https://github.com/alanqrwang/keymorph/releases).
Download your preferred model(s) and put them in the folder specified by `--weights_dir` in the commands below.

## Registering brain volumes 
### BrainMorph
WARNING: Please see the [BrainMorph repository](https://github.com/alanqrwang/brainmorph) for the latest updates and models! This is a legacy version of the code and is not guaranteed to be maintained.

BrainMorph is trained on over 100,000 brain MR images at full resolution (256x256x256). 
The script will automatically min-max normalize the images and resample to 1mm isotropic resolution.

`--num_keypoints` and `num_levels_for_unet` will determine which model will be used to perform the registration.
Make sure the corresponding weights are present in `--weights_dir`.
`--num_keypoints` can be set to `128, 256, 512` and `--num_levels_for_unet` can be set to `4, 5, 6, 7`, respectively (corresponding to 'S', 'M', 'L', 'H' in the paper).

To register a single pair of volumes:
```
python scripts/register.py \
    --num_keypoints 256 \
    --num_levels_for_unet 4 \
    --weights_dir ./weights/ \
    --moving ./example_data/img_m/IXI_000001_0000.nii.gz \
    --fixed ./example_data/img_m/IXI_000002_0000.nii.gz \
    --moving_seg ./example_data/seg_m/IXI_000001_0000.nii.gz \
    --fixed_seg ./example_data/seg_m/IXI_000002_0000.nii.gz \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize
```

Description of other important flags:
+ `--moving` and `--fixed` are paths to moving and fixed images.
+ `--moving_seg` and `--fixed_seg` are optional, but are required if you want the script to report Dice scores. 
+ `--list_of_aligns` specifies the types of alignment to perform. Options are `rigid`, `affine` and `tps_<lambda>` (TPS with hyperparameter value equal to lambda). lambda=0 corresponds to exact keypoint alignment. lambda=10 is very similar to affine.
+ `--list_of_metrics` specifies the metrics to report. Options are `mse`, `harddice`, `softdice`, `hausd`, `jdstd`, `jdlessthan0`. To compute Dice scores and surface distances, `--moving_seg` and `--fixed_seg` must be provided.
+ `--save_eval_to_disk` saves all outputs to disk. The default location is `./register_output/`.
+ `--visualize` plots a matplotlib figure of moving, fixed, and registered images overlaid with corresponding points.

You can also replace filenames with directories to register all images in the directory.
Note that the script expects corresponding image and segmentation pairs to have the same filename.
```bash
python scripts/register.py \
    --num_keypoints 256 \
    --num_levels_for_unet 4 \
    --weights_dir ./weights/ \
    --moving ./example_data/img_m/ \
    --fixed ./example_data/img_m/ \
    --moving_seg ./example_data/seg_m/ \
    --fixed_seg ./example_data/seg_m/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize
```

### Groupwise registration
```bash
python scripts/register.py \
    --groupwise \
    --num_keypoints 256 \
    --num_levels_for_unet 4 \
    --weights_dir ./weights/ \
    --moving ./example_data/ \
    --fixed ./example_data/ \
    --moving_seg ./example_data/ \
    --fixed_seg ./example_data/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize
```



### IXI-trained, half-resolution models
All other model weights are trained on half-resolution (128x128x128) on the (smaller) IXI dataset. 
The script will automatically min-max normalize the images.
To register two volumes with our best-performing model:

```bash
python scripts/register.py \
    --half_resolution \
    --num_keypoints 512 \
    --backbone conv \
    --moving ./example_data_half/img_m/IXI_001_128x128x128.nii.gz \
    --fixed ./example_data_half/img_m/IXI_002_128x128x128.nii.gz \
    --load_path ./weights/numkey512_tps0_dice.4760.h5 \
    --moving_seg ./example_data_half/seg_m/IXI_001_128x128x128.nii.gz \
    --fixed_seg ./example_data_half/seg_m/IXI_002_128x128x128.nii.gz \
    --list_of_aligns affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --visualize
```

## Step-by-step guide for reproducing our results

### Dataset 
[A] Scripts in `./notebooks/[A] Download Data` will download the IXI data and perform some basic preprocessing

[B] Once the data is downloaded `./notebooks/[B] Brain extraction` can be used to extract remove non-brain tissue. 

[C] Once the brain has been extracted, we center the brain using `./notebooks/[C] Centering`. During training, we randomly introduce affine augmentation to the dataset. This ensure that the brain stays within the volume given the affine augmentation we introduce. It also helps during the pretraining step of our algorithm.

### Pretraining KeyMorph

This step helps with the convergence of our model. We pick 1 subject and random points within the brain of that subject. We then introduce affine transformation to the subject brain and same transformation to the keypoints. In other words, this is a self-supervised task in where the network learns to predict the keypoints on a brain under random affine transformation. We found that initializing our model with these weights helps with the training.

To pretrain, run:
 
```bash
python scripts/run.py \
    --run_mode pretrain \
    --num_keypoints 128 \
    --loss_fn mse \
    --transform_type tps_0 \
    --max_random_affine_augment_params (0.2, 0.2, 3.1416, 0.1) \
    --affine_slope 1000 \
    --data_dir ./centered_IXI 
```

`--affine_slope` linearly ramps up the ``max_random_affine_augment_params` such that it starts at 0 for all parameters and reaches their maximum values at epoch 1000. This helps the model to learn the keypoints under increasing affine transformations.

### Training KeyMorph
Follow instructions for "Training KeyMorph" above, for more options.

```bash
python scripts/run.py \
    --run_mode train \
    --num_keypoints 128 \
    --loss_fn mse \
    --transform_type affine \
    --train_dataset ixi \
    --data_path ./centered_IXI \
    --max_random_affine_augment_params (0.2, 0.2, 3.1416, 0.1) \
    --load_path ./weights/numkey128_pretrain.2500.h5 
```

`--load_path <path>` specifies the path to the pretraining weights.

### Evaluating KeyMorph
```bash
python scripts/run.py \
    --run_mode eval \
    --num_keypoints 128 \
    --loss_fn dice \
    --transform_type tps_0 \
    --data_dir ./centered_IXI \
    --load_path ./weights/best_trained_model.h5 \
    --save_eval_to_disk
```

## Related Projects
+ For evaluation, we use [SynthSeg](https://github.com/BBillot/SynthSeg) to automatically segment different brain regions. Follow their repository for detailed intruction on how to use the model. 
+ [BrainMorph](https://github.com/alanqrwang/brainmorph) is a foundation model based on the KeyMorph framework, trained on over 100,000 brain MR images at full resolution (256x256x256).
The model is robust to normal and diseased brains, a variety of MRI modalities, and skullstripped and non-skullstripped images.
Check out the dedicated repository for the latest updates and models!

## Issues
This repository is being actively maintained. Feel free to open an issue for any problems or questions.

## Legacy code
For a legacy version of the code, see our [legacy branch](https://github.com/alanqrwang/keymorph/tree/legacy).

## References
If this code is useful to you, please consider citing our papers.
The first conference paper contains the unsupervised, affine version of KeyMorph.
The second, follow-up journal paper contains the unsupervised/supervised, affine/TPS versions of KeyMorph.

Evan M. Yu, et al. "[KeyMorph: Robust Multi-modal Affine Registration via Unsupervised Keypoint Detection.](https://openreview.net/forum?id=OrNzjERFybh)" (MIDL 2021).

Alan Q. Wang, et al. "[A Robust and Interpretable Deep Learning Framework for Multi-modal Registration via Keypoints.](https://arxiv.org/abs/2304.09941)" (Medical Image Analysis 2023).
