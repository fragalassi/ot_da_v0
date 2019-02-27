import os
import glob
import sys

sys.path.append('/udd/aackaouy/OT-DA/')

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators, get_validation_split
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model

class Train_Isensee:
    def __init__(self, conf):
        self.config = conf

        
    def fetch_training_data_files(self, return_subject_ids=False):
        training_data_files = list()
        subject_ids = list()
        for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "../Data/data_"+self.config.data_set, "training", "*")):
            subject_ids.append(os.path.basename(subject_dir))
            subject_files = list()
            for modality in self.config.training_modalities + ["./"+self.config.GT]:  # Autre solution ? "/ManualSegmentation/ pour miccai16"
                subject_files.append(os.path.join(subject_dir, modality + ".nii.gz")) # + "/Preprocessed/ pour miccai16
            training_data_files.append(tuple(subject_files))
        if return_subject_ids:
            return training_data_files, subject_ids
        else:
            return training_data_files


    def main(self, overwrite_data=True, overwrite_model = True):
        # convert input images into an hdf5 file
        if overwrite_data or not os.path.exists(self.config.data_file):
            training_files, subject_ids = self.fetch_training_data_files(return_subject_ids=True)
            write_data_to_file(training_files, self.config.data_file, image_shape=self.config.image_shape,
                               subject_ids=subject_ids)
        else:
            print("Reusing previously written data file. Set overwrite_data to True to overwrite this file.")

        data_file_opened = open_data_file(self.config.data_file)

        if not overwrite_model and os.path.exists(self.config.model_file):
            model = load_old_model(self.config.model_file)
        else:
            # instantiate new model

            model = isensee2017_model(input_shape=self.config.input_shape, n_labels=self.config.n_labels,
                                      initial_learning_rate=self.config.initial_learning_rate,
                                      n_base_filters=self.config.n_base_filters, loss_function=self.config.loss_function,
                                      shortcut=self.config.shortcut)

        # get training and testing generators
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            data_file_opened,
            batch_size=self.config.batch_size,
            data_split=self.config.validation_split,
            overwrite_data=overwrite_data,
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


        # run training
        train_model(model=model,
                    model_file=self.config.model_file,
                    training_generator=train_generator,
                    validation_generator=validation_generator,
                    steps_per_epoch=n_train_steps,
                    validation_steps=n_validation_steps,
                    initial_learning_rate=self.config.initial_learning_rate,
                    learning_rate_drop=self.config.learning_rate_drop,
                    learning_rate_patience=self.config.patience,
                    early_stopping_patience=self.config.early_stop,
                    n_epochs=self.config.epochs,
                    niseko=self.config.niseko)

        data_file_opened.close()
