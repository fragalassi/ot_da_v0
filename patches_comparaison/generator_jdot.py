import os
import copy
from random import shuffle
import itertools

import numpy as np
import sys
from unet3d.utils import pickle_dump, pickle_load
from unet3d.utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from unet3d.augment import augment_data, random_permutation_x_y
from multiprocessing.pool import Pool
from time import time
import random



def get_batch_jdot(selected_source, selected_target, source_data_file, target_data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite_data=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_overlap = 0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False, number_of_threads = 64,
                                           target = True, validation = False, source_center = ["01"], target_center = ["07"], all = False):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    source_x, source_y = data_generator_jdot_multi_proc(selected_source,
                                        source_data_file,
                                        validation=validation,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=training_patch_overlap,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        shuffle_index_list=True,
                                        permute=permute,
                                        number_of_threads = number_of_threads,
                                        all = all,
                                        )
    if target:
        target_x, target_y = data_generator_jdot_multi_proc(selected_target,
                                            target_data_file,
                                            validation=validation,
                                            batch_size=batch_size,
                                            n_labels=n_labels,
                                            labels=labels,
                                            augment=augment,
                                            augment_flip=augment_flip,
                                            augment_distortion_factor=augment_distortion_factor,
                                            patch_shape=patch_shape,
                                            patch_overlap=training_patch_overlap,
                                            patch_start_offset=training_patch_start_offset,
                                            skip_blank=skip_blank,
                                            shuffle_index_list=True,
                                            permute=permute,
                                            number_of_threads = number_of_threads,
                                            all = all,
                                            )

        x = np.vstack((source_x, target_x))
        y = np.vstack((source_y, target_y))
        training_batch = (x,y)


    else:
        training_batch = (source_x, source_y)

    return training_batch


def get_patches_index_list(source_data_file, target_data_file, training_keys_file_source, validation_keys_file_source,
                           training_keys_file_target, validation_keys_file_target, source_center,
                           target_center, data_split = 0.8, change_validation = True, patch_shape = 16, skip_blank = True,
                           training_patch_overlap = 0.5, validation_patch_overlap = 0.5, training_patch_start_offset = None):

    source_training_list, source_validation_list = get_validation_split(source_data_file,
                                                          data_split=data_split,
                                                          change_validation=change_validation,
                                                          training_file=training_keys_file_source,
                                                          validation_file=validation_keys_file_source,)

    source_training_path = os.path.abspath("Data/generated_data/training_list_gt_"+source_center)
    source_validation_path = os.path.abspath("Data/generated_data/validation_list_gt_" + source_center)



    if skip_blank:

        save_patches_with_gt(source_training_list, source_data_file, patch_shape, training_patch_overlap,
                             training_patch_start_offset, path=source_training_path, overwrite = change_validation)

        source_training_list = load_index_patches_with_gt(source_training_path)

        save_patches_with_gt(source_validation_list, source_data_file, patch_shape, validation_patch_overlap,
                             training_patch_start_offset, path=source_validation_path, overwrite = change_validation)

        source_validation_list = load_index_patches_with_gt(source_validation_path)

    target_training_path = os.path.abspath("Data/generated_data/training_list_gt_"+target_center)
    target_validation_path = os.path.abspath("Data/generated_data/validation_list_gt_" + target_center)

    target_training_list, target_validation_list = get_validation_split(target_data_file,
                                                          data_split=data_split,
                                                          change_validation=change_validation,
                                                          training_file=training_keys_file_target,
                                                          validation_file=validation_keys_file_target)
    if skip_blank:
        save_patches_with_gt(target_training_list, target_data_file, patch_shape, training_patch_overlap,
                             training_patch_start_offset, path=target_training_path, overwrite = change_validation)
        target_training_list = load_index_patches_with_gt(target_training_path)

        save_patches_with_gt(target_validation_list, target_data_file, patch_shape, validation_patch_overlap,
                             training_patch_start_offset, path=target_validation_path, overwrite = change_validation)
        target_validation_list = load_index_patches_with_gt(target_validation_path)

    return source_training_list, source_validation_list, target_training_list, target_validation_list

def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

def get_validation_split(data_file, training_file, validation_file, data_split=0.8, change_validation=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if change_validation or not os.path.exists(training_file):
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def data_generator_jdot_multi_proc(selected, data_file, validation = True, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False, number_of_threads = 64, all = False):
    '''
    Create a batch for jdot:
    source data: x_list[:batch_size]
    target data: y_list[batch_size:]

    :param data_file:
    :param index_list:
    :param batch_size:
    :param n_labels:
    :param labels:
    :param augment:
    :param augment_flip:
    :param augment_distortion_factor:
    :param patch_shape:
    :param patch_overlap:
    :param patch_start_offset:
    :param shuffle_index_list:
    :param skip_blank:
    :param permute:
    :param number_of_threads: Parameter to set the max number of threads used for the loading
    :return:
    '''
    training_x_list = []
    training_y_list = []
    training_x_list, training_y_list,  = multi_proc_loop(selected, data_file, training_x_list, training_y_list, batch_size = batch_size,
                                         stopping_criterion= batch_size, number_of_threads = number_of_threads,
                                         patch_shape=patch_shape, augment=augment, augment_flip=augment_flip,
                                         augment_distortion_factor=augment_distortion_factor, skip_blank=skip_blank,
                                         permute=permute, all = all)

    training_x_list, training_y_list = convert_data(training_x_list, training_y_list, n_labels=n_labels, labels=labels)

    return training_x_list, training_y_list

def multi_proc_loop(index_list, data_file, x_list, y_list, batch_size = 64, stopping_criterion = 64,
                    number_of_threads = 64, patch_shape = 16, augment = False, augment_flip = False,
                    augment_distortion_factor = None, skip_blank = False, permute = False, all = False):
    '''
    The loop for loading data with multiprocess.
    :param index_list:
    :param data_file:
    :param x_list:
    :param y_list:
    :param batch_size:
    :param stopping_criterion:
    :param number_of_threads:
    :param patch_shape:
    :param augment:
    :param augment_flip:
    :param augment_distortion_factor:
    :param skip_blank:
    :param permute:
    :return:
    '''
    remaining_batch = batch_size
    selected_index = []
    while len(index_list) > 0:
        print(len(index_list))
        # Two verifications for the remaining samples to put in the batch.
        # We want set the number_of_threads to the number of samples remaining
        if len(index_list) > number_of_threads:
            n = number_of_threads
        else:
            n = len(index_list)
        pool = Pool(number_of_threads)
        results = []
        for i in range(n):
            index = index_list.pop()
            selected_index += [index]
            data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
            if patch_shape is not None:
                affine = data_file.root.affine[index[0]]
            else:
                affine = data_file.root.affine[index]
            results.append(pool.apply_async(add_data_mp, args=(data, truth, affine, index, augment, augment_flip,
                                                                   augment_distortion_factor, patch_shape, skip_blank,
                                                                   permute)))
        pool.close()
        pool.join()
        results = [r.get() for r in results]


        for i in range(len(results)):
            if len(results[i][0]) != 0 and (remaining_batch != 0 or all):
                remaining_batch -= 1
                x_list.append(results[i][0][0])
                y_list.append(results[i][1][0])
        end = time()
        if len(x_list) == stopping_criterion or (len(index_list) == 0 and len(x_list) > 0):
            break
    return x_list, y_list

def save_patches_with_gt(index_list, data_file, patch_shape, patch_overlap, patch_start_offset, path, overwrite):
    if not os.path.exists(path) or overwrite:
        print("Creating and saving a file containing the index of patches with GT. This may take a while...")
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape,
                                                      patch_overlap, patch_start_offset)
        index_list = get_patches_with_ground_truth(index_list, data_file, patch_shape)
        pickle_dump(index_list, path)

def load_index_patches_with_gt(file_name):
    return pickle_load(file_name)

def get_patches_with_ground_truth(index_list, data_file, patch_shape):
    new_index_list = []
    initial_length = len(index_list)
    while len(index_list) > 0:
        advance = "\r Writing indexes with GT :" + str((initial_length - len(index_list))/initial_length*100 )+ "%"
        sys.stdout.write(advance)
        sys.stdout.flush()
        index = index_list.pop()
        data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
        if np.mean(truth) != 0:
            new_index_list += [index]
    return new_index_list

def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    if augment:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)


def add_data_mp(data, truth, affine, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    x_list = []
    y_list = []
    if augment:
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)

    return (x_list, y_list)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
