from training_testing import create_test
import os
from scipy.spatial.distance import cdist, dice, cosine, euclidean, jaccard, braycurtis
from scipy.spatial import minkowski_distance
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import  get_data_from_file, get_validation_split, create_patch_index_list, add_data
from unet3d.utils.patches import get_patch_from_3d_data
from unet3d.utils.utils import pickle_load
from unet3d.metrics import dice_coef_loss
import numpy as np
import pandas as pd
import random
import nibabel as nib
import itertools
import pickle
import sys
from unet3d.model import isensee2017_model
from patches_comparaison.JDOT import JDOT


class Activations:

    def __init__(self, conf):
        self.config = conf
        create = create_test.Test(self.config)
        self.fetch_testing_data_files = create.fetch_testing_data_files
        self.patch_shape = (16, 16, 16)
        self.jd = None

    def main(self):
        self.load_model()
        index_list, validation_list, data_file = self.get_index_list()
        index_list, data_file = self.get_index_list_GT()


    def load_model(self):
        model, context_output_name = isensee2017_model(input_shape=self.config.input_shape, n_labels=self.config.n_labels,
                                      initial_learning_rate=self.config.initial_learning_rate,
                                      n_base_filters=self.config.n_base_filters,
                                      loss_function=self.config.loss_function,
                                      shortcut=self.config.shortcut,
                                      compile=False)

        jd = JDOT(model, config=self.config, context_output_name=context_output_name)
        jd.load_old_model(self.config.model_file)
        jd.compile_model()
        self.jd = jd

    def compute_activation(self, x_a, x_b):
        x_a = np.expand_dims(x_a, axis=0)
        x_b = np.expand_dims(x_b, axis=0)
        x_a = self.jd.model.predict(x_a)
        x_b = self.jd.model.predict(x_b)
        return x_a[-2][0], x_b[-2][0]

