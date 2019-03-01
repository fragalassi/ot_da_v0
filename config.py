import os

'''
Config class definition, here are grouped all the parameters.

setting test to false in main.py result in using only two examples and a reduce amout of epochs

'''

class Config:
    def __init__(self, test=False, rev = 0, batch_size = 1, initial_lr = 5e-4, loss_function = "generalized_dice_loss", depth = 5,
                 n_filter=16, niseko=True, shortcut=True):
        if test == True:
            self.data_set="test"
            self.epochs = 1
        else:
            self.data_set="miccai16_preprocessed"
            self.epochs = 1000  # cutoff the training after this many epochs

        self.niseko = niseko

        self.rev = int(rev)
        print(self.rev)

        self.image_shape = (128,128,128)  # This determines what shape the images will be cropped/resampled to.
        self.patch_shape = (16,16,16)  # switch to None to train on the whole image

        self.shortcut = shortcut  # If True, the architecture will be using shortcuts

        self.labels=(1)
        self.n_labels=1
        self.all_modalities=["FLAIR-norm-include", "T1-norm-include"]
        self.GT = "Consensus-reg-m-include"
        self.training_modalities= self.all_modalities  # change this if you want to only use some of the modalities
        self.nb_channels = len(self.training_modalities)

        if self.patch_shape is not None:
            self.input_shape = tuple([self.nb_channels] + list(self.patch_shape))
        else:
            self.input_shape = tuple([self.nb_channels] + list(self.image_shape))

        self.batch_size = int(float(batch_size))
        self.validation_batch_size = self.batch_size
        self.loss_function = loss_function

        self.patience = 20  # learning rate will be reduced after this many epochs if the validation loss is not improving
        self.early_stop = 100  # training will be stopped after this many epochs without the validation loss improving
        self.initial_learning_rate = float(initial_lr)
        self.learning_rate_drop = 0.5  # factor by which the learning rate will be reduced
        self.validation_split = 0.8  # portion of the data that will be used for training

        self.flip = False  # augments the data by randomly flipping an axis during
        self.permute = False  # data shape must be a cube. Augments the data by permuting in various directions
        self.distort = None  # switch to None if you want no distortion
        self.augment = self.permute or self.distort
        self.validation_patch_overlap = 0  # if > 0, during training, validation patches will be overlapping
        self.training_patch_start_offset = None #(16,16,16)  # randomly offset the first patch index by up to this offset
        self.skip_blank = True  # if True, then patches without any target will be skipped

        self.overwrite_data = True  # If True, will previous files. If False, will use previously written files.
        self.overwrite_model = True

        self.data_file = os.path.abspath("Data/generated_data/"+self.data_set+"_data.h5")

        if not os.path.exists(os.path.abspath("Data/generated_data")):
            os.makedirs(os.path.abspath("Data/generated_data"))

        self.n_base_filters = int(float(n_filter))
        self.depth = int(float(depth))
        self.model_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_2017_model_rev"+str(self.rev)+".h5")
        self.training_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_training_ids.pkl")
        self.validation_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_validation_ids.pkl")


