# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

from training_testing import train_isensee2017, predict, evaluate, create_test
from patches_comparaison import train_jdot
from activation_prediction import activation_prediction
from unet3d.data import write_data_to_file, open_data_file
import config
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
from patches_comparaison import compare_patches
from keras import backend as K
from Config import create_config
from multiprocessing.pool import ThreadPool
import os


sys.path.append('/udd/aackaouy/OT-DA/')

# df = create_config.create_conf(batch_size_l = [32, 64], initial_lr_l = [5e-2, 5e-4],
#                  loss_funcs = ["dice_coefficient_loss", "dice_coefficient_loss"],
#                  depth_l = [5, 8], n_filters=[8, 16, 32], patch_shape_l=[16, 32], overlap_l=[0, 0.5],  n_exp = 30)

'''
Best configuration yet.
Need to be tested with data augmentation.
'''

batch_size = [125]*3
initial_lr = [5e-2]*3
loss_funcs = ["dice_coefficient_loss"]*3
depth = [5]*3
n_filter = [16]*3
patch_shape = [16]*3
overlap = [1/2]*3
image_shape = [(128,128,128)]*3
training_center = [["All"]]*3
augmentation = [True]*3
jdot_alpha = [0.001]*3
bool_train_jdot = [True]*3
source_center = ["01", "08", "08"]
target_center = ["08", "01", "07"]
alpha_factor = [1]*3
df = create_config.create_conf_with_l(batch_size, initial_lr, loss_funcs,
                                      depth, n_filter, patch_shape, overlap, training_center,
                                      image_shape, augmentation, jdot_alpha, source_center, target_center,
                                      bool_train_jdot, alpha_factor,
                                      n_repeat=1)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)

for i in range(df.shape[0]): #df.shape[0]
    print("Experience number:", i+1)
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=False, rev=i, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i],
                         n_filter=df["Number of filters"].iloc[i],
                         patch_shape = df["Patch shape"].iloc[i],
                         overlap = df["Overlap"].iloc[i],
                         augmentation = df["Augmentation"].iloc[i],
                         jdot_alpha=df["JDOT Alpha"].iloc[i],
                         source_center=df["Source center"].iloc[i],
                         target_center=df["Target center"].iloc[i],
                         training_centers = df["Training centers"].iloc[i],
                         image_shape = df["Image shape"].iloc[i],
                         bool_train_jdot = df["Train JDOT"].iloc[i],
                         alpha_factor = df["Alpha factor"].iloc[i],
                         niseko=True, shortcut=True)

    '''
    To compare patches
    '''
    # comp = compare_patches.Compare_patches(conf)
    # comp.compute_activations()
    # comp.main()

    # conf.all_modalities = ["FLAIR-include"]
    # data_file_opened = open_data_file(os.path.abspath("Data/generated_data/" + conf.data_set + "_data_source.h5"))
    # comp.save_patch(3, np.array([60, 20, 28]), "A", data_file_opened, 0)
    # comp.save_patch(3, np.array([44, 36, 84]), "B", data_file_opened, 0)
    '''
    For JDOT, uncomment this part
    '''

    train_jd = train_jdot.Train_JDOT(conf)
    train_jd.main(overwrite_data=conf.overwrite_data, overwrite_model=conf.overwrite_model)

    test = create_test.Test(conf)
    test.main(overwrite_data=conf.overwrite_data)

    eval = evaluate.Evaluate(conf)
    eval.main()

    '''
    For normal training uncomment this part
    '''

    # train = train_isensee2017.Train_Isensee(conf)
    # train.main(overwrite_data=conf.overwrite_data, overwrite_model=conf.overwrite_model)
    #
    # test = create_test.Test(conf)
    # test.main(overwrite_data=conf.overwrite_data)
    #
    # pred = predict.Predict(conf)
    # pred.main()
    #
    # eval = evaluate.Evaluate(conf)
    # eval.main()

    K.clear_session()
