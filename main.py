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
from patches_comparaison import compare_patches, intensities
from keras import backend as K
from Config import create_config
from scipy.spatial.distance import cdist
from multiprocessing.pool import ThreadPool
import os
import argparse

sys.path.append('/udd/aackaouy/OT-DA/')

# df = create_config.create_conf(batch_size_l = [32, 64], initial_lr_l = [5e-2, 5e-4],
#                  loss_funcs = ["dice_coefficient_loss", "dice_coefficient_loss"],
#                  depth_l = [5, 8], n_filters=[8, 16, 32], patch_shape_l=[16, 32], overlap_l=[0, 0.5],  n_exp = 30)

'''
Best configuration yet.
Need to be tested with data augmentation.
'''

parser = argparse.ArgumentParser()
parser.add_argument("-source", type=str, help="Set the source center")
parser.add_argument("-target", type=str, help="Set the target center")
parser.add_argument("-alpha", type=float, help="Set JDOT alpha")
parser.add_argument("-beta", type=float, help="Set JDOT beta")
parser.add_argument("-jdot", type=str, help="Bool to train on JDOT")
parser.add_argument("-shape", type=str, help="Patch shape")
parser.add_argument("-augment", type=str, help="Boolean for data augmentation")
parser.add_argument("-rev", type=int, help="The id of the revision")
parser.add_argument("-epochs", type=int, help="Number of epochs")
parser.add_argument("-lr", type=float, help="Set the initial lr")
parser.add_argument("-callback", type=str, help="Boolean for the usage of callback")
parser.add_argument("-dist", type=str, help="Distance used to compute the Optimal Transport. Can be sqeuclidean or dice.")
parser.add_argument("-OT_depth", type=int, help="Depth to compute the OT on. 5 is the most compact. 9 is the deepest.")
parser.add_argument("-load_model", type=str, help="Wether to load the base model or not")
parser.add_argument("-force_training_list", type=str, help="Tuple of tuples, first level is for source/target, second level is for training/validation")


args = parser.parse_args()

batch_size = [256]
initial_lr = [args.lr]
loss_funcs = ["dice_coefficient_loss"]
depth = [5]
n_filter = [16]
patch_shape = [args.shape]
training_overlap = [0]
testing_overlap = [1/2]
image_shape = [(128,128,128)]
force_training_list = [eval(args.force_training_list)]
training_center = [["All"]]
augmentation = [True if args.augment == "True" else False]
jdot_alpha = [args.alpha]
jdot_beta = [args.beta]
bool_train_jdot = [True if args.jdot == "True" else False]
source_center = [args.source]
target_center = [args.target]
epochs = [args.epochs]
callback = [True if args.callback == "True" else False]
alpha_factor = [1]
distance = [args.dist]
OT_depth = [args.OT_depth]
load_model = [args.load_model]

df = create_config.create_conf_with_l(batch_size, initial_lr, loss_funcs,
                                      depth, n_filter, patch_shape, training_overlap, testing_overlap, training_center,
                                      image_shape, augmentation, jdot_alpha, source_center, target_center,
                                      bool_train_jdot, alpha_factor, epochs, callback, distance, OT_depth,
                                      jdot_beta, load_model, force_training_list,
                                      n_repeat=1)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)

for i in range(df.shape[0]): #df.shape[0]
    print("Experience number:", i+1)
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=False, rev=args.rev, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i],
                         n_filter=df["Number of filters"].iloc[i],
                         patch_shape = df["Patch shape"].iloc[i],
                         training_overlap = df["Training overlap"].iloc[i],
                         testing_overlap = df["Testing overlap"],
                         augmentation = df["Augmentation"].iloc[i],
                         jdot_alpha=df["JDOT Alpha"].iloc[i],
                         source_center=df["Source center"].iloc[i],
                         target_center=df["Target center"].iloc[i],
                         training_centers = df["Training centers"].iloc[i],
                         image_shape = df["Image shape"].iloc[i],
                         bool_train_jdot = df["Train JDOT"].iloc[i],
                         alpha_factor = df["Alpha factor"].iloc[i],
                         epochs = df["Epochs"].iloc[i],
                         callback = df["Callback"].iloc[i],
                         distance = df["Distance"].iloc[i],
                         OT_depth = df["OT Depth"].iloc[i],
                         jdot_beta = df["JDOT beta"].iloc[i],
                         load_model = df["Load model"].iloc[i],
                         force_training_list = df["Force training list"].iloc[i],
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

    # comp = intensities.Compute_intensities(conf)
    # comp.fetch_training_data_files()
    # comp.compute_intensity()
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
