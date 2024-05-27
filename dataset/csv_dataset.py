import csv
import os
import torchio as tio
from dataset.utils import (
    KeyMorphDataset,
)


class CSVDataset(KeyMorphDataset):
    def __init__(self, csv_file) -> None:
        super().__init__()
        self.csv_file = csv_file

    def _get_subjects_dict(self, train):
        subjects_dict = {}
        total_subjects = 0
        self.seg_available = False

        with open(self.csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if (row["train"].lower() == "true") == train:
                    modality = row["modality"]

                    if modality not in subjects_dict:
                        subjects_dict[modality] = []

                    subject = tio.Subject(
                        img=tio.ScalarImage(os.path.join(row["img_path"])),
                    )
                    if os.path.join(row["seg_path"]) != "None":
                        self.seg_available = True
                        subject.add_image(
                            tio.LabelMap(os.path.join(row["seg_path"])), "seg"
                        )
                    if os.path.join(row["mask_path"]) != "None":
                        subject.add_image(
                            tio.LabelMap(os.path.join(row["mask_path"])), "mask"
                        )

                    subjects_dict[modality].append(subject)
                    total_subjects += 1

        # Print statistics
        print(f"\nSplit train={train}")
        print(f"Total number of subjects: {total_subjects}")
        for modality, subjects in subjects_dict.items():
            print(f"Modality: {modality}, Number of subjects: {len(subjects)}")
        return subjects_dict
