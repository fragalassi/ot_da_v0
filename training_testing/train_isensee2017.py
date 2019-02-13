import os
import glob
import sys

sys.path.append('/temp_dd/igrida-fs1/aackaouy/3D-Unet/')

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators, get_validation_split
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model

test = True

config = dict()

if test == True:
    config["data_set"] = "test"
else:
    config["data_set"] = "miccai16"

config["rev"] = 0

config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
#config["image_shape"] = (144, 144, 144)  
#config["image_shape"] = (176,176,176) 

config["patch_shape"] = None  # switch to None to train on the whole image
# config["patch_shape"] = (176,176,176)

config["labels"] = (1)
#config["n_labels"] = len(config["labels"])
config["n_labels"] = 1

config["all_modalities"] = ["Preprocessed/FLAIR_preprocessed", "Preprocessed/T1_preprocessed"]

config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

config["batch_size"] = 1
config["validation_batch_size"] = 2

if test == True:
    config["n_epochs"] = 1  # cutoff the training after this many epochs
else:
    config["n_epochs"] = 500

config["patience"] = 20  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 100  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training

config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped
config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.

config["data_file"] = os.path.abspath("Data/"+config["data_set"]+"_data.h5")

config["n_base_filters"] = 16
config["depth"] = 5
config["model_file"] = os.path.abspath("Data/"+config["data_set"]+"_isensee_2017_model_rev"+str(config["rev"])+".h5") #  patch (128,128,128) ; n_filters = 16; ski_blank = True; depth = 5
config["training_file"] = os.path.abspath("Data/"+config["data_set"]+"_isensee_training_ids_rev"+str(config["rev"])+".pkl")
config["validation_file"] = os.path.abspath("Data/"+config["data_set"]+"_isensee_validation_ids_rev"+str(config["rev"])+".pkl")

#config["n_base_filters"] = 16
#config["depth"] = 5
#config["model_file"] = os.path.abspath("miccai16_isensee_2017_model_all_rev0.h5") #  (176,176,176)
#config["training_file"] = os.path.abspath("miccai16_isensee_training_ids_all_rev0.pkl")
#config["validation_file"] = os.path.abspath("miccai16_isensee_validation_ids_all_rev0.pkl")
#config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.
##config["crop_slices"] = None

#config["crop"] = True
#config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.
#
##--------- START FG: USE THIS TO CREATE TESTING DATASET ! ------------------
#config["validation_split"] = 0.0  # portion of the data that will be used for training
#config["data_file"] = os.path.abspath("testing_FG_rev3.h5")
#config["training_file"] = os.path.abspath("testing_training_FG_ids_rev3.pkl")
#config["validation_file"] = os.path.abspath("testing_FG_validation_ids_rev3.pkl")
#
#def fetch_training_data_files(return_subject_ids=False):
#    training_data_files = list()
#    subject_ids = list()
#    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data_miccai16","testing_FG", "*")):
##            for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "brain_set_4", "*")):
#        subject_ids.append(os.path.basename(subject_dir))
#        subject_files = list()
##        for modality in config["training_modalities"] + ["T2-norm-include"]:
#        for modality in config["training_modalities"] + ["Consensus-reg-m-include"]:
#            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
#        training_data_files.append(tuple(subject_files))
#        
#    if return_subject_ids:
#        return training_data_files, subject_ids
#    else:
#        return training_data_files
#
#def main(overwrite=True):
#    # convert input images into an hdf5 file
#    if overwrite or not os.path.exists(config["data_file"]):
#        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)
#
#        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
#                           subject_ids=subject_ids)
#    data_file_opened = open_data_file(config["data_file"])
#    # get training and testing generators
#    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
#        data_file_opened,
#        batch_size=config["batch_size"],
#        data_split=config["validation_split"],
#        overwrite=overwrite,
#        validation_keys_file=config["validation_file"],
#        training_keys_file=config["training_file"],
#        n_labels=config["n_labels"],
#        labels=config["labels"],
#        patch_shape=config["patch_shape"],
#        validation_batch_size=config["validation_batch_size"],
#        validation_patch_overlap=config["validation_patch_overlap"],
#        training_patch_start_offset=config["training_patch_start_offset"],
#        permute=config["permute"],
#        augment=config["augment"],
#        skip_blank=config["skip_blank"],
#        augment_flip=config["flip"],
#        augment_distortion_factor=config["distort"])
#
#    
#if __name__ == "__main__":
#    main(overwrite=config["overwrite"])
##--------- END FG: USE THIS TO CREATE TESTING DATASET ! ------------------

        
def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "Data/data_"+config["data_set"], "training", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["ManualSegmentation/Consensus"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)
                           
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()
