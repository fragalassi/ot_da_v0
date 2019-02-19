from training_testing import train_isensee2017, predict, evaluate, create_test
import config
import sys
import pandas as pd
import numpy as np
from Config import create_config

sys.path.append('/udd/aackaouy/OT-DA/')

# df = create_config.create_conf(batch_size_l = [1, 1], initial_lr_l = [1e-4, 1e-7],
#                  loss_funcs = ["generalized_dice_loss", "weighted_dice_coefficient_loss"],
#                  depth_l = [3, 8], n_exp = 3)

batch_size = [1,1,1]
initial_lr = [5e-4,5e-4,5e-4]
loss_funcs = ["weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss", "weighted_dice_coefficient_loss"]
depth = [4,5,6]

df = create_config.create_conf_with_l(batch_size, initial_lr, loss_funcs, depth)

print(df)

for i in range(df.shape[0]):
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=True, rev=i, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i])

    train = train_isensee2017.Train_Isensee(conf)
    train.main(overwrite=conf.overwrite)

    test = create_test.Test(conf)
    test.main(overwrite=conf.overwrite)

    pred = predict.Predict(conf)
    pred.main()

    eval = evaluate.Evaluate(conf)
    eval.main()
