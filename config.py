import os
from keras.optimizers import Adam
'''
Config class definition, here are grouped all the parameters.

setting test to false in main.py result in using only two examples and a reduce amout of epochs

'''

class Config:
    def __init__(self, test=False, rev = 0, batch_size = 1, initial_lr = 5e-4, loss_function = "generalized_dice_loss", depth = 5,
                 n_filter=16, patch_shape = 16, training_overlap = None, testing_overlap = None, training_centers=["All"],
                 image_shape = (128,128,128) , niseko=True, shortcut=True, augmentation=False,
                 jdot_alpha = 0.001, jdot_beta = 0.0001, source_center = ["01"], bool_train_jdot = True, target_center = ["07"],
                 alpha_factor = 1, epochs = 1000, callback = False, distance="sqeuclidean", OT_depth = 5, load_model=False, force_training_list = (([0,1,2,4], [3]),([0,1,2,4], [3]))):
        '''

        :param test: To only use the test data with only two training cases and 3 testing cases
        :param rev: The number given to the files to test different configurations
        :param batch_size:
        :param initial_lr:
        :param loss_function:
        :param depth:
        :param n_filter:
        :param patch_shape:
        :param niseko: When working on niseko CPU workers need to be set to 0
        :param shortcut: To test the 3D-Unet with no shortcuts (spoil it doesn't work well)
        '''

        if test == True:
            self.data_set="test"
            self.epochs = 1
            self.all_modalities = ["FLAIR-include"]
        else:
            self.data_set="miccai16_no_norm"
            self.epochs = epochs  # cutoff the training after this many epochs
            self.all_modalities = ["FLAIR-include", "T1-include"]

        self.niseko = niseko

        self.rev = int(rev)
        print("Revision :", self.rev)
        self.one_patient = False

        self.image_shape = image_shape# This determines what shape the images will be cropped/resampled to.
        self.patch_shape = (int(float(patch_shape)),int(float(patch_shape)),int(float(patch_shape)))  # switch to None to train on the whole image

        self.shortcut = shortcut  # If True, the architecture will be using shortcuts
        self.training_centers = training_centers
        self.source_center = source_center
        self.target_center = target_center

        self.number_of_threads = 64
        self.load_all_data = True  # Parameter to load all the data directly in memory. Set it to false if you don't have much ram available.

        self.labels=(1)
        self.n_labels=1 # Number of labels JDOT never was tested with more than one parameter.

        # self.all_modalities = ["FLAIR-norm-include"]
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
        self.optimizer = Adam

        self.callback = callback
        self.patience = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
        self.early_stop = 20  # training will be stopped after this many epochs without the validation loss improving
        self.initial_learning_rate = float(initial_lr)
        self.learning_rate_drop = 0.5  # factor by which the learning rate will be reduced
        self.validation_split = 0.8  # portion of the data that will be used for training

        self.train_jdot = bool_train_jdot
        self.jdot_alpha = jdot_alpha
        self.jdot_beta = jdot_beta
        self.alpha_factor = alpha_factor
        self.depth_jdot = OT_depth # 5 is the middle layer (with the smallest representation) 9 is the deepest layer
        self.jdot_distance = distance #Distance used for the computation of gamma (sqeuclidean or dice)

        '''
        If augmentation is set to true, both flip and permutation transforms are taken into account.
        '''
        self.flip = augmentation  # augments the data by randomly flipping an axis during
        self.permute = augmentation  # data shape must be a cube. Augments the data by permuting in various directions
        self.distort = None  # switch to None if you want no distortion
        self.augment = self.permute or self.distort
        self.validation_patch_overlap = int(float(testing_overlap)*float(patch_shape))  # if > 0, during training, validation patches will be overlapping
        self.training_patch_overlap = int(float(training_overlap)*float(patch_shape))  # Overlap could be the number of overlapping pixels.
        self.training_patch_start_offset = None #(16,16,16)  # randomly offset the first patch index by up to this offset
        self.skip_blank = False  # if True, then patches without any target will be skipped
        self.save_image = False

        self.overwrite_data = True # If True, will previous files. If False, will use previously written files.
        self.change_validation = True
        self.force_training_list = force_training_list
        self.overwrite_model = True
        self.load_base_model = load_model


        self.n_base_filters = int(float(n_filter))
        self.depth = int(float(depth))

        '''
        Definition of all the path to the files that will be generated. They all lie in the Data/generated_data folder.
        '''
        if not os.path.exists(os.path.abspath("Data/generated_data")):
            os.makedirs(os.path.abspath("Data/generated_data"))

        self.data_file = os.path.abspath("Data/generated_data/"+self.data_set+"_data.h5") # Training without JDOT: one single h5 where the data is saved.
        self.source_data_file = os.path.abspath("Data/generated_data/"+self.data_set+"_data_center_"+str(source_center)+".h5") # Source data file for a given center
        self.target_data_file = os.path.abspath("Data/generated_data/"+self.data_set+"_data_center_"+str(target_center)+".h5") # Target data file for a given center

        self.model_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_2017_model_rev"+str(self.rev)+".h5") # The model that will be created (or loaded)

        self.training_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_training_ids.pkl")
        self.validation_file = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_validation_ids.pkl")

        # Next are all the files containing the IDs of a given validation split.
        self.training_file_source = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_training_ids_center_"+str(self.source_center)+".pkl")
        self.validation_file_source = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_validation_ids_center_"+str(self.source_center)+".pkl")
        self.training_file_target = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_training_ids_center_"+str(self.target_center)+".pkl")
        self.validation_file_target = os.path.abspath("Data/generated_data/"+self.data_set+"_isensee_validation_ids_center_"+str(self.target_center)+".pkl")

        # The directories where the results will be saved.
        self.save_dir =  os.path.abspath("results/prediction/rev_" + str(self.rev))
        self.prediction_dir = os.path.abspath("results/prediction/rev_" + str(self.rev) + "/prediction_" + self.data_set)
