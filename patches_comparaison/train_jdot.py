import os
import glob
import sys

sys.path.append('/udd/aackaouy/OT-DA/')

from unet3d.data import write_data_to_file, open_data_file
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
from patches_comparaison.JDOT import JDOT

class Train_JDOT:
    def __init__(self, conf):
        self.config = conf

    def fetch_training_data_files(self, return_subject_ids=False):
        '''
        Function to get the training files from the source and from the target.
        We write the source and target samples in two different files
        :param return_subject_ids:
        :return:
        '''
        source_data_files = list()
        target_data_files = list()
        subject_ids_source = list()
        subject_ids_target = list()
        for subject_dir in glob.glob(
                os.path.join(os.path.dirname(__file__), "../Data/data_" + self.config.data_set, "training", "*")):
            subject_center = subject_dir[-9:-7]  # Retrieve for the MICCAI16 data-set the center of the patient

            if subject_center in self.config.source_center or self.config.source_center == ["All"]:
                subject_ids_source.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in self.config.training_modalities + [
                    "./" + self.config.GT]:  # Autre solution ? "/ManualSegmentation/ pour miccai16"
                    subject_files.append(
                        os.path.join(subject_dir, modality + ".nii.gz"))  # + "/Preprocessed/ pour miccai16
                source_data_files.append(tuple(subject_files))

            if subject_center in self.config.target_center or self.config.target_center == ["All"]:
                subject_ids_target.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in self.config.training_modalities + [
                    "./" + self.config.GT]:  # Autre solution ? "/ManualSegmentation/ pour miccai16"
                    subject_files.append(
                        os.path.join(subject_dir, modality + ".nii.gz"))  # + "/Preprocessed/ pour miccai16
                target_data_files.append(tuple(subject_files))
        if return_subject_ids:
            return source_data_files, target_data_files, subject_ids_source, subject_ids_target
        else:
            return source_data_files, target_data_files

    def main(self, overwrite_data=True, overwrite_model=True):
        # convert input images into an hdf5 file
        if overwrite_data or not os.path.exists(self.config.data_file):
            source_data_files, target_data_files, subject_ids_source, subject_ids_target = self.fetch_training_data_files(return_subject_ids=True)
            write_data_to_file(source_data_files, self.config.source_data_file, image_shape=self.config.image_shape,
                               subject_ids=subject_ids_source)
            write_data_to_file(target_data_files, self.config.target_data_file, image_shape=self.config.image_shape,
                               subject_ids=subject_ids_target)
        else:
            print("Reusing previously written data file. Set overwrite_data to True to overwrite this file.")

        source_data = open_data_file(self.config.source_data_file)
        target_data = open_data_file(self.config.target_data_file)

        if not overwrite_model and os.path.exists(self.config.model_file):
            model = load_old_model(self.config.model_file)
        else:
            # instantiate new model

            model = isensee2017_model(input_shape=self.config.input_shape, n_labels=self.config.n_labels,
                                      initial_learning_rate=self.config.initial_learning_rate,
                                      n_base_filters=self.config.n_base_filters,
                                      loss_function=self.config.loss_function,
                                      shortcut=self.config.shortcut,
                                      compile=False)
        # get training and testing generators

        jd = JDOT(model, self.config, source_data, target_data)
        jd.compile_model()
        jd.train_model(21)
        jd.evaluate_model()


        # run training
        # train_model(model=model,
        #             model_file=self.config.model_file,
        #             training_generator=train_generator,
        #             validation_generator=validation_generator,
        #             steps_per_epoch=n_train_steps,
        #             validation_steps=n_validation_steps,
        #             initial_learning_rate=self.config.initial_learning_rate,
        #             learning_rate_drop=self.config.learning_rate_drop,
        #             learning_rate_patience=self.config.patience,
        #             early_stopping_patience=self.config.early_stop,
        #             n_epochs=self.config.epochs,
        #             niseko=self.config.niseko)

        source_data.close()
        target_data.close()

