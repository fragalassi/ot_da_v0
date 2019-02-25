from training_testing import train_isensee2017, predict, evaluate, create_test
import config
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
from keras import backend as K
from Config import create_config

sys.path.append('/udd/aackaouy/OT-DA/')

df = create_config.create_conf(batch_size_l = [2, 2], initial_lr_l = [1e-4, 1e-7],
                 loss_funcs = ["weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss"],
                 depth_l = [3, 8], n_filters=[8, 32],  n_exp = 30)

# batch_size = [2,2,2,2,2,2,2,2,2]
# initial_lr = [5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4]
# loss_funcs = ["weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss"]
# depth = [3,3,3,4,4,4,6,6,6] # 5 16 on a déjà
# n_filter = [16, 16, 16, 16, 16, 16, 16, 16, 16]
#
# df = create_config.create_conf_with_l(batch_size, initial_lr, loss_funcs, depth, n_filter)

print(df)

for i in range(df.shape[0]):
    print("Experience number:", i+1)
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=True, rev=i+9, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i],
                         n_filter=df["Number of filters"].iloc[i],
                         niseko=False)

    train = train_isensee2017.Train_Isensee(conf)
    train.main(overwrite_data=conf.overwrite_data, overwrite_model=conf.overwrite_model)

    test = create_test.Test(conf)
    test.main(overwrite_data=conf.overwrite_data)

    pred = predict.Predict(conf)
    pred.main()

    eval = evaluate.Evaluate(conf)
    eval.main()

    K.clear_session()
