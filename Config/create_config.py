import pandas as pd
from pyDOE import lhs
from math import ceil
import numpy as np
import os
def create_conf(batch_size_l = [1, 8], initial_lr_l = [1e-4, 1e-7],
                loss_funcs = ["generalized_dice_loss", "weighted_dice_coefficient_loss"],
                depth_l = [3, 8], n_filters=[8, 32], n_exp=4, n_repeat = 2):
    batch_size = []
    initial_lr = []
    loss_func = []
    depth = []


    l = lhs(4, samples=n_exp)

    for exp in l:
        batch_size += [ceil((batch_size_l[1] - batch_size_l[0]) * exp[0]) + batch_size_l[0]]
        initial_lr += [(initial_lr_l[0] - initial_lr_l[1]) * exp[1] + initial_lr_l[1]]
        loss_func += [loss_funcs[0] if exp[2]<=0.5 else loss_funcs[1]]
        depth += [ceil((depth_l[1] - depth_l[0]) * exp[3] + depth_l[0])]

    print("Tested parameters")
    print("===========")
    df = pd.DataFrame(np.array([batch_size, initial_lr, loss_func, depth])).T
    df.columns = ["Batch Size", "Initial Learning Rate", "Loss function", "Depth"]
    print(df)
    print("===========")

    save_path = os.path.abspath("Config/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path+"/Config.csv")
    df.to_csv(path_or_buf=file_name, sep=";")

    return df

def create_conf_with_l(batch_size=[], initial_lr = [], loss_funcs = [], depth=[], n_filter=[]):
    lists = [batch_size, initial_lr, loss_funcs, depth]
    it = iter(lists)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Not all lists have same length')
    else:
        df = pd.DataFrame(np.array([batch_size, initial_lr, loss_funcs, depth, n_filter])).T
        df.columns = ["Batch Size", "Initial Learning Rate", "Loss function", "Depth", "Number of filters"]
    save_path = os.path.abspath("Config/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path + "/Config.csv")
    df.to_csv(path_or_buf=file_name, sep=";")
    return df