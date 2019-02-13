# -*- coding: utf-8 -*-
"""
Loss functions for multi-class segmentation
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot
    

def generalized_dice_loss(ground_truth, prediction, weight_map = None, type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
        
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
         tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
#    generalised_dice_denominator = tf.reduce_sum(
#        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    del ref_vol,seg_vol,generalised_dice_denominator,weights,new_weights
                             
    return -generalised_dice_score
    
def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in

        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016

    using a square in the denominator

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(
            tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    # dice_score.set_shape([num_classes])
    # minimising (1 - dice_coefficients)
    return -tf.reduce_mean(dice_score)
    
    
    
def sensitivity_specificity(prediction,
                                 ground_truth,
                                 weight_map=None,
                                 r=0.05):
    """
    Function to calculate a multiple-ground_truth version of
    the sensitivity-specificity loss defined in "Deep Convolutional
    Encoder Networks for Multiple Sclerosis Lesion Segmentation",
    Brosch et al, MICCAI 2015,
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1

    error is the sum of r(specificity part) and (1-r)(sensitivity part)

    :param prediction: the logits
    :param ground_truth: segmentation ground_truth.
    :param r: the 'sensitivity ratio'
        (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    prediction = tf.cast(prediction, tf.float32)
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # value of unity everywhere except for the previous 'hot' locations
    one_cold = 1 - one_hot

    # chosen region may contain no voxels of a given label. Prevents nans.
    epsilon = 1e-5

    squared_error = tf.square(one_hot - prediction)
    specificity_part = tf.reduce_sum(
        squared_error * one_hot, 0) / \
                       (tf.reduce_sum(one_hot, 0) + epsilon)
    sensitivity_part = \
        (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) /
         (tf.reduce_sum(one_cold, 0) + epsilon))

    return tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)    