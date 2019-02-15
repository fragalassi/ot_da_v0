from training_testing import train_isensee2017, predict, evaluate, create_test
import config
import sys
import pandas as pd
import numpy as np
import os
import tables
from math import ceil
from pyDOE import lhs

sys.path.append('/udd/aackaouy/OT-DA/')

batch_size = []
initial_lr = []
loss_func = []

batch_size_l = [1, 8]
initial_lr_l = [1e-4, 1e-7]
loss_funcs = ["generalized_dice_loss", "weighted_dice_coefficient_loss"]
loss_ceil = 1 / len(loss_funcs)

l = lhs(3, samples=5)

for exp in l:
    batch_size += [ceil((batch_size_l[1] - batch_size_l[0]) * exp[0])]
    initial_lr += [(initial_lr_l[0] - initial_lr_l[1]) * exp[1]]
    loss_func += [loss_funcs[0] if exp[2]<=0.5 else loss_funcs[1]]

print("Tested parameters")
print("===========")
df = pd.DataFrame(np.array([batch_size, initial_lr, loss_func])).T
df.columns = ["Batch Size", "Initial Learning Rate", "Loss function"]
print(df)
print("===========")

save_path = os.path.abspath("Config/")
if not os.path.exists(save_path):
    os.makedirs(save_path)
file_name = os.path.join(save_path+"/Config.csv")
df.to_csv(path_or_buf=file_name, sep=";")

for i in range(df.shape[0]):

    conf = config.Config(test=False, rev=i, batch_size=df["Batch Size"].iloc[i], initial_lr=df["Initial Learning Rate"].iloc[i], loss_function=df["Loss function"].iloc[i])

    train = train_isensee2017.Train_Isensee(conf)
    train.main(overwrite=conf.overwrite)

    test = create_test.Test(conf)
    test.main(overwrite=conf.overwrite)

    pred = predict.Predict(conf)
    pred.main()

    eval = evaluate.Evaluate(conf)
    eval.main()
