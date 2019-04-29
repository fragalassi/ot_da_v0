from unet3d.generator import  get_data_from_file
import glob
import os
from unet3d.data import write_data_to_file, open_data_file

class Compute_intensities:

    def __init__(self, conf):
        self.config = conf

    def fetch_training_data_files(self):
        ids = list()
        data_files = list()
        for subject_dir in glob.glob(
                os.path.join(os.path.dirname(__file__), "../Data/data_" + self.config.data_set, "training", "*")):
            subject_center = subject_dir[-9:-7]  # Retrieve for the MICCAI16 data-set the center of the patient

        ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in self.config.training_modalities + [
            "./" + self.config.GT]:  # Autre solution ? "/ManualSegmentation/ pour miccai16"
            subject_files.append(
                os.path.join(subject_dir, modality + ".nii.gz"))  # + "/Preprocessed/ pour miccai16
        data_files.append(tuple(subject_files))
        write_data_to_file(data_files, self.config.source_data_file, image_shape=self.config.image_shape,
                           subject_ids=ids)

    def compute_intensity(self):
        data_file = open_data_file(self.config.source_data_file)
        get_data_from_file(data_file, l[0], self.patch_shape)

