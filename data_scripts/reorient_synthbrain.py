import os
import subprocess
from pathlib import Path
import torchio as tio
from keymorph import utils
import numpy as np
from pprint import pprint
from argparse import ArgumentParser

# WARNING!!!!
# On AI cluster, load FSL and bc modules.
# Activate conda environment alw4013-hdbet.


def shell_command(command):
    print("RUNNING", command)
    subprocess.run(command, shell=True)


def _reorient(src_dir, tgt_dir):
    """Reorient to MNI space using fslreorient2std"""
    failed = []
    for basename in sorted(os.listdir(src_dir)):
        try:
            name = basename.split(".")[0]
            s = os.path.join(src_dir, basename)
            t = os.path.join(tgt_dir, name)
            reorient_command = f"fslreorient2std {s} {t}"
            shell_command(reorient_command)

        except Exception as e:
            print("Error reorienting:", src_dir)
            failed.append(src_dir)

    return failed


def main():
    all_failed = []
    base_dir = Path("/midtier/sablab/scratch/alw4013/data/synthseg_clean/")

    reorient_dir = Path("/midtier/sablab/scratch/alw4013/data/synthseg_clean_MNI")

    # fslreorient2std
    ds_src_img_dir = base_dir / "image"
    ds_src_seg_dir = base_dir / "labels"
    ds_tgt_img_dir = reorient_dir / "image"
    ds_tgt_seg_dir = reorient_dir / "labels"

    if not os.path.exists(ds_tgt_img_dir):
        os.makedirs(ds_tgt_img_dir)
    if not os.path.exists(ds_tgt_seg_dir):
        os.makedirs(ds_tgt_seg_dir)

    failed = _reorient(
        ds_src_img_dir,
        ds_tgt_img_dir,
    )
    all_failed += failed

    failed = _reorient(
        ds_src_seg_dir,
        ds_tgt_seg_dir,
    )
    all_failed += failed

    if len(all_failed) == 0:
        print("All successful!")
    else:
        print("Failed on:\n")
        pprint(all_failed)


if __name__ == "__main__":
    main()
