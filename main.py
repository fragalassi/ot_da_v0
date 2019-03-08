# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

from training_testing import train_isensee2017, predict, evaluate, create_test
from activation_prediction import activation_prediction
import config
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
from patches_comparaison import compare_patches
from keras import backend as K
from Config import create_config


sys.path.append('/udd/aackaouy/OT-DA/')

df = create_config.create_conf(batch_size_l = [32, 64], initial_lr_l = [5e-2, 5e-4],
                 loss_funcs = ["dice_coefficient_loss", "dice_coefficient_loss"],
                 depth_l = [3, 8], n_filters=[8, 16, 32], patch_shape_l=[8, 16, 32], overlap_l=[0, 0.5],  n_exp = 30)

# batch_size = [64, 64, 64, 64, 64, 64, 64, 32, 64, 64]
# initial_lr = [5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3]
# loss_funcs = ["dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss", "dice_coefficient_loss"]
# depth = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# n_filter = [16, 16, 16, 8, 16, 16, 16, 16, 16, 16]
# patch_shape = [32, 32, 32, 32, 8, 16, 32, 32, 32, 32]
# overlap = [1/2, 1/4, 1/10, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 0]
# image_shape = [(128,128,128), (128,128,128), (128,128,128), (128,128,128), (128,128,128), (128,128,128), (128,128,128), (176,176,176), (256,256,176), (128,128,128)]
# training_center = [["All"], ["All"], ["All"], ["All"], ["All"], ["All"], ["All"], ["All"], ["All"], ["All"]]
# df = create_config.create_conf_with_l(batch_size, initial_lr, loss_funcs,
#                                       depth, n_filter, patch_shape, overlap, training_center,
#                                       image_shape,
#                                       n_repeat=1)
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)

for i in range(df.shape[0]): #df.shape[0]
    print("Experience number:", i+1)
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=True, rev=i, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i],
                         n_filter=df["Number of filters"].iloc[i],
                         patch_shape = df["Patch shape"].iloc[i],
                         overlap = df["Overlap"].iloc[i],
                         # training_centers = df["Training centers"].iloc[i],
                         # image_shape = df["Image shape"].iloc[i],
                         niseko=True, shortcut=True)

    # patch_comparaison = compare_patches.Compare_patches(conf)
    # patch_comparaison.main()

    train = train_isensee2017.Train_Isensee(conf)
    train.main(overwrite_data=conf.overwrite_data, overwrite_model=conf.overwrite_model)

    test = create_test.Test(conf)
    test.main(overwrite_data=conf.overwrite_data)

    pred = predict.Predict(conf)
    pred.main()

    eval = evaluate.Evaluate(conf)
    eval.main()

    K.clear_session()
