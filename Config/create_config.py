import pandas as pd
from pyDOE import lhs
from math import ceil
import numpy as np
import os
def create_conf(batch_size_l = [1, 8], initial_lr_l = [1e-4, 1e-7],
                loss_funcs = ["generalized_dice_loss", "weighted_dice_coefficient_loss"],
                depth_l = [3, 8], n_filters=[8, 32], n_exp=4, n_repeat = 2):
    '''
    A function to create following the Latin Hypersquare experience plan configurations to train the network.
    :param batch_size_l:
    :param initial_lr_l:
    :param loss_funcs:
    :param depth_l:
    :param n_filters:
    :param n_exp: Number of experience we want.
    :param n_repeat:
    :return:
    '''
    batch_size = []
    initial_lr = []
    loss_func = []
    depth = []
    n_filter = []


    l = lhs(5, samples=n_exp) # Create an array of experiences giving pseudo random number (latin hypersquare) number between 0 and 1.

    for exp in l:
        batch_size += [round((batch_size_l[1] - batch_size_l[0]) * exp[0]) + batch_size_l[0]]
        initial_lr += [(initial_lr_l[0] - initial_lr_l[1]) * exp[1] + initial_lr_l[1]]
        loss_func += [loss_funcs[0] if exp[2]<=0.5 else loss_funcs[1]]
        depth += [round((depth_l[1] - depth_l[0]) * exp[3] + depth_l[0])]
        n_filter += [round((n_filters[1] - n_filters[0]) * exp[4] + n_filters[0])]

    #Creating all the different configurations from the

    print("Tested parameters")
    print("===========")
    df = pd.DataFrame(np.array([batch_size, initial_lr, loss_func, depth, n_filter])).T
    df.columns = ["Batch Size", "Initial Learning Rate", "Loss function", "Depth", "Number of filters"]
    print("===========")

    save_path = os.path.abspath("Config/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path+"/Config.csv")
    df.to_csv(path_or_buf=file_name, sep=";")

    return df

def create_conf_with_l(batch_size=[], initial_lr = [], loss_funcs = [], depth=[], n_filter=[], patch_shape = [], overlap = [], training_center = [], n_repeat = 1):
    '''
    Create configuration from the list.
    :param batch_size:
    :param initial_lr:
    :param loss_funcs:
    :param depth:
    :param n_filter:
    :param patch_shape:
    :param n_repeat: The number of times we want each experience to be repeated
    :return: A data-frame with all the experiences
    '''
    lists = [batch_size, initial_lr, loss_funcs, depth, patch_shape, overlap, training_center]
    it = iter(lists)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Not all lists have same length')
    else:
        df = pd.DataFrame(np.array([batch_size*n_repeat, initial_lr*n_repeat, loss_funcs*n_repeat,
                                    depth*n_repeat, n_filter*n_repeat, patch_shape*n_repeat, overlap*n_repeat, training_center*n_repeat])).T
        df.columns = ["Batch Size", "Initial Learning Rate", "Loss function", "Depth", "Number of filters", "Patch shape", "Overlap", "Training centers"]
    save_path = os.path.abspath("Config/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path + "/Config.csv")
    df.to_csv(path_or_buf=file_name, sep=";")
    return df