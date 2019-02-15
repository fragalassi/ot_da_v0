import os
import glob
import sys

sys.path.append('/udd/aackaouy/OT-DA/')

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators, get_validation_split
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model

class Test:
    def __init__(self, conf):
        self.config = conf

    def fetch_testing_data_files(self, return_subject_ids=False):
        testing_data_files = list()
        subject_ids = list()
        for subject_dir in glob.glob(
                os.path.join(os.path.dirname(__file__), "../Data/data_" + self.config.data_set, "testing", "*")):
            #            for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "brain_set_4", "*")):
            subject_ids.append(os.path.basename(subject_dir))
            subject_files = list()
            #        for modality in config["training_modalities"] + ["T2-norm-include"]:
            for modality in self.config.training_modalities + ["./"+self.config.GT]: #"/ManualSegmentation/"
                subject_files.append(os.path.join(subject_dir+"/", modality + ".nii.gz"))
            testing_data_files.append(tuple(subject_files))

        if return_subject_ids:
            return testing_data_files, subject_ids
        else:
            return testing_data_files


    def main(self, overwrite=True):
        self.config.validation_split = 0.0
        self.config.data_file = os.path.abspath("Data/generated_data/" + self.config.data_set + "_testing_rev_" + str(self.config.rev) + ".h5")
        self.config.training_file = os.path.abspath(
            "Data/generated_data/" + self.config.data_set + "_testing_rev" + str(self.config.rev) + ".pkl")
        self.config.validation_file = os.path.abspath(
            "Data/generated_data/" + self.config.data_set + "_testing_validation_ids_rev" + str(self.config.rev) + ".pkl")
        # convert input images into an hdf5 file
        if overwrite or not os.path.exists(self.config.data_file):
            testing_files, subject_ids = self.fetch_testing_data_files(return_subject_ids=True)
            print(testing_files)
            write_data_to_file(testing_files, self.config.data_file, image_shape=self.config.image_shape,
                               subject_ids=subject_ids)
        data_file_opened = open_data_file(self.config.data_file)
        # get training and testing generators
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            data_file_opened,
            batch_size=self.config.batch_size,
            data_split=self.config.validation_split,
            overwrite=overwrite,
            validation_keys_file=self.config.validation_file,
            training_keys_file=self.config.training_file,
            n_labels=self.config.n_labels,
            labels=self.config.labels,
            patch_shape=self.config.patch_shape,
            validation_batch_size=self.config.validation_batch_size,
            validation_patch_overlap=self.config.validation_patch_overlap,
            training_patch_start_offset=self.config.training_patch_start_offset,
            permute=self.config.permute,
            augment=self.config.augment,
            skip_blank=self.config.skip_blank,
            augment_flip=self.config.flip,
            augment_distortion_factor=self.config.distort)

        data_file_opened.close()