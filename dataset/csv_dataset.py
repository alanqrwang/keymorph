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

    def _has_modality_header(self, file_path):
        with open(file_path, mode="r") as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the first line, which should be the header
            return "modality" in headers

    def get_subjects(self, train):
        if self._has_modality_header(self.csv_file):
            return self._get_subjects_dict(train=train)
        else:
            return self._get_subjects_two_lists(train=train)

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
                        modality=modality,
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

    def _get_subjects_two_lists(self, train):
        fixed_subjects = []
        moving_subjects = []
        total_fixed_subjects = 0
        total_moving_subjects = 0
        self.seg_available = False

        with open(self.csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if (row["train"].lower() == "true") == train:
                    # Create fixed subject
                    fixed_subject = tio.Subject(
                        img=tio.ScalarImage(os.path.join(row["fixed_img_path"])),
                        modality="fixed",
                    )
                    if row["fixed_seg_path"] != "None":
                        self.seg_available = True
                        fixed_subject.add_image(
                            tio.LabelMap(os.path.join(row["fixed_seg_path"])), "seg"
                        )
                    if row["fixed_mask_path"] != "None":
                        fixed_subject.add_image(
                            tio.LabelMap(os.path.join(row["fixed_mask_path"])), "mask"
                        )
                    fixed_subjects.append(fixed_subject)
                    total_fixed_subjects += 1

                    # Create moving subject
                    moving_subject = tio.Subject(
                        img=tio.ScalarImage(os.path.join(row["moving_img_path"])),
                        modality="moving",
                    )
                    if row["moving_seg_path"] != "None":
                        self.seg_available = True
                        moving_subject.add_image(
                            tio.LabelMap(os.path.join(row["moving_seg_path"])), "seg"
                        )
                    if row["moving_mask_path"] != "None":
                        moving_subject.add_image(
                            tio.LabelMap(os.path.join(row["moving_mask_path"])), "mask"
                        )
                    moving_subjects.append(moving_subject)
                    total_moving_subjects += 1

        # Print statistics
        print(f"\nSplit train={train}")
        print(f"Total number of fixed subjects: {total_fixed_subjects}")
        print(f"Total number of moving subjects: {total_moving_subjects}")

        return fixed_subjects, moving_subjects
