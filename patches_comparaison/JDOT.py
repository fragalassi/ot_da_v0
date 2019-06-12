from keras import backend as K
from keras import Model
from keras.callbacks import LambdaCallback
from patches_comparaison.generator_jdot import get_batch_jdot, multi_proc_augment_data, get_patches_index_list
from scipy.spatial.distance import cdist, cosine, euclidean, dice
from unet3d.utils import pickle_load
import tables
from unet3d.prediction import run_validation_case
import os
import numpy as np
from numpy import all
import ot
from training_testing import create_test
from keras.models import load_model
import time
import random
import sys
from copy import copy
from multiprocessing import Pool

class JDOT():

    def __init__(self, model, config = None, source_data = None , target_data = None, context_output_name = None, allign_loss=1.0, tar_cl_loss=1.0,
                 sloss=0.0, tloss=1.0, int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):


        self.config = config
        self.model = model  # target model
        if self.config.depth_jdot:
            self.context_output_name = [context_output_name[self.config.depth_jdot -1]]
        else:
            self.context_output_name = []
        self.source_data = source_data
        self.target_data = target_data

        self.batch_size = self.config.batch_size
        self.optimizer = self.config.optimizer
        self.jdot_alpha = K.variable(self.config.jdot_alpha)
        self.jdot_beta = K.variable(self.config.jdot_beta)
        # initialize the gamma (coupling in OT) with zeros
        self.gamma = K.zeros(shape=(self.batch_size, self.batch_size))
        self.batch_source = K.zeros(shape=(self.batch_size,
                                    len(self.config.training_modalities),self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))
        self.batch_target = K.zeros(shape=(self.batch_size,
                                    len(self.config.training_modalities),self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))

        self.image_representation_source = []
        self.image_representation_target = []
        # whether to minimize with classification loss
        self.train_cl = K.variable(tar_cl_loss)
        # whether to minimize with the allignment loss
        self.train_algn = K.variable(allign_loss)
        self.sloss = K.variable(sloss)  # weight for source classification
        self.tloss = K.variable(tloss)  # weight for target classification
        self.verbose = verbose
        self.int_lr = int_lr  # initial learning rate
        self.lr_decay = lr_decay
        #
        self.ot_method = ot_method

        self.train_batch = ()
        self.validation_batch = ()

        self.complete_source_training_list = []
        self.complete_source_validation_list = []
        self.complete_target_training_list = []
        self.complete_target_validation_list = []

        self.source_training_list = []
        self.source_validation_list = []
        self.target_training_list = []
        self.target_validation_list = []
        self.epoch_complete = False
        self.validation_complete = False

        self.training_data = None
        self.validation_data = None
        self.affine_source_training = None
        self.affine_source_validation = None
        self.affine_target_training = None
        self.affine_target_validation = None

        self.t = 0
        self.count = 0

        self.prediction = []

        self.target_pred = K.zeros(shape=(self.batch_size, 1, self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))
        self.source_truth = K.zeros(shape=(self.batch_size, 1, self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))

        def deep_jdot_loss_euclidean(y_true, y_pred):
            '''
            Segmentation alignment loss function for the squared euclidean distance.
            Source loss is the dice coefficient loss for the source samples and their labels. It avoids a too big degradation of the results in the source domain.
            Target loss is the pairwise squared euclidean distance between source labels and prediction on the target domain.

            :param y_true:
            :param y_pred:
            :return: Source_loss + \beta * \gamma * target_loss
            The more two samples are connected by \gamma the more we wish they have similar predictions.
            '''
            truth_source = y_true[:self.batch_size, :]  # source true labels
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :] # target prediction

            '''
            Compute the loss function of the source samples.
            '''
            source_loss = dice_coefficient_loss(truth_source, prediction_source)
            target_loss = euclidean_dist(K.batch_flatten(truth_source), K.batch_flatten(prediction_target))
            return source_loss + self.jdot_beta * K.sum(self.gamma * target_loss)
        self.deep_jdot_loss_euclidean = deep_jdot_loss_euclidean

        def deep_jdot_loss_dice(y_true, y_pred):
            '''
            Same for the dice "distance".
            :param y_true:
            :param y_pred:
            :return:
            '''
            truth_source = y_true[:self.batch_size, :]  # source true labels
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :] # target prediction
            '''
            Compute the loss function of the source samples.
            '''
            source_loss = dice_coefficient_loss(truth_source, prediction_source)
            target_loss = parwise_dice_coefficient(K.batch_flatten(truth_source), K.batch_flatten(prediction_target))
            return source_loss + self.jdot_alpha * K.sum(self.gamma * target_loss)
        self.deep_jdot_loss_dice = deep_jdot_loss_dice

        distance_dic = {
            "sqeuclidean": deep_jdot_loss_euclidean,
            "dice": deep_jdot_loss_dice,
        }

        self.deep_jdot_loss = distance_dic[self.config.jdot_distance]
        print("Using ", self.config.jdot_distance, "distance to compute gamma")


        def distance_loss(y_true, y_pred):
            '''
            Representation alignement loss function.
            Dif is the pairwise euclidean distance between the source samples and the target samples.

            :param y_true:
            :param y_pred:
            :return: \alpha * sum(\gamma*dif)
            The more two samples are connected by gamma (source and target) the more the representation should be similar.
            '''
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :]  # target prediction
            dif = euclidean_dist(K.batch_flatten(prediction_source), K.batch_flatten(prediction_target))
            return self.jdot_alpha * K.sum(self.gamma*dif)

        self.distance_loss = distance_loss

        def jdot_image_loss(y_true, y_pred):
            '''
            Custom jdot_loss function.
            :param y_true:
            :param y_pred:
            :return: A sum of the source_loss and the OT loss.
            '''
            truth_source = y_true[:self.batch_size, :]  # source true labels
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :]  # target prediction

            '''
            Compute the loss function of the source samples.
            '''
            source_loss = dice_coefficient_loss(truth_source, prediction_source)

            '''
            Compute the euclidean distance between each of the source samples and each of the target samples.
            It returns a matrix (batch_size, batch_size).
            This euclidean distance is computed both in the image space and in the truth/prediction space.
            '''
            # cos_distance_samples = cos_distance(K.batch_flatten(self.batch_source),K.batch_flatten(self.batch_target))
            # cos_distance_pred = cos_distance(K.batch_flatten(truth_source), K.batch_flatten(prediction_target))

            euc_distance_samples = euclidean_dist(K.batch_flatten(self.batch_source),K.batch_flatten(self.batch_target))
            euc_distance_pred = euclidean_dist(K.batch_flatten(truth_source), K.batch_flatten(prediction_target))
            return source_loss + self.jdot_alpha*K.sum(self.gamma*(K.abs(euc_distance_samples - euc_distance_pred)))

        self.jdot_image_loss = jdot_image_loss

        def euclidean_dist(x,y):
            """
            Pairwise euclidean distance.
            :param x:
            :param y:
            :return: A matrix of size n_batch*n_batch where each entry represent the euclidean distance between one source sample and one target sample.
            """
            dist = K.reshape(K.sum(K.square(x), 1), (-1, 1))
            dist += K.reshape(K.sum(K.square(y), 1), (1, -1))
            dist -= 2.0*K.dot(x, K.transpose(y))

            return K.sqrt(dist)


        def dice_coefficient(y_true, y_pred, smooth=1.):
            '''
            Dice coefficient score.
            :param y_true:
            :param y_pred:
            :param smooth:
            :return: The Dice Coefficient between the vector of prediction for all the batch and the vector of true for all the batch. It's equivalent to a mean on each unique
            vector.
            '''
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient = dice_coefficient

        def parwise_dice_coefficient(y_true, y_pred, smooth=1.):
            '''
            Pairwise dice coefficient.
            :param y_true:
            :param y_pred:
            :param smooth:
            :return: A matrix of size n_batch*n_batch where each entry represent the Dice Score between one source sample and one target sample.
            '''
            x = K.expand_dims(y_true, axis=-1)
            y = y_pred
            intersection = x * K.transpose(y)
            intersection = K.sum(K.permute_dimensions(intersection, (0, 2, 1)), axis = -1)
            sum = K.abs(x) + K.abs(K.transpose(y))
            sum = K.sum(K.permute_dimensions(sum, (0, 2, 1)), axis = -1)
            return (2 * intersection + smooth)/(sum + smooth)
        self.pairwise_dice_coefficient = parwise_dice_coefficient

        def dice_coefficient_target(y_true, y_pred, smooth=1.):
            '''
            Dice coefficient on the sample for the target domain. Useful when some labels are available in the target domain to monitor the improvement due to the adaptation.
            However in the full unsupervised domain adaptation problem, we don't have access to this information.
            :param y_true:
            :param y_pred:
            :param smooth:
            :return:
            '''
            y_true_f = K.flatten(y_true[self.batch_size:, :])
            y_pred_f = K.flatten(y_pred[self.batch_size:, :])
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient_target = dice_coefficient_target

        def dice_coefficient_source(y_true, y_pred, smooth=1.):
            '''
            Dice coefficient on the sample for the source domain. Useful when some labels are available in the target domain to monitor the improvement due to the adaptation.
            However in the full unsupervised domain adaptation problem, we don't have access to this information.
            :param y_true:
            :param y_pred:
            :param smooth:
            :return:
            '''

            y_true_f = K.flatten(y_true[:self.batch_size, :])
            y_pred_f = K.flatten(y_pred[:self.batch_size, :])
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient_source = dice_coefficient_source


        def dice_coefficient_loss(y_true, y_pred):
            '''
            Dice coefficient loss function.
            :param y_true:
            :param y_pred:
            :return:
            '''
            return 1-dice_coefficient(y_true, y_pred)
        self.dice_coefficient_loss = dice_coefficient_loss




        '''
        Uncomment to check if these functions are computing the right values.
        '''

        # x = np.array([[0, 1, 0], [200, 0, 1], [1, 1, 0]])
        # y = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 1]])
        # z = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        # print(1-dice(x[0],y[0]))
        # print(1-dice(x[1], y[1]))
        # print(1-dice(x[2],y[2]))
        # print("Evaluation: \n", K.eval(dice_coefficient(K.constant(x), K.constant(y))))
        # print("Evaluation pairwise: \n", K.eval(parwise_dice_coefficient(K.constant(x), K.constant(y))))


    def compile_model(self):
        '''
        Compilation with the custom loss function and the metrics.
        :return:
        '''
        if self.config.train_jdot:
            if self.config.depth_jdot == None:
                self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate, clipnorm=1., clipvalue=0.5), loss=self.jdot_image_loss, metrics=[self.dice_coefficient, self.dice_coefficient_source, self.dice_coefficient_target])
            else:
                outputs = [self.model.get_layer(name).output for name in self.context_output_name]
                outputs += [self.model.layers[-1].output]
                self.model = Model(inputs=self.model.input,
                                                 outputs=outputs)
                loss = {
                    self.model.layers[-1].name : self.deep_jdot_loss,
                    self.context_output_name[-1]: self.distance_loss,
                }
                self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate, clipnorm=1., clipvalue=0.5), loss=loss, metrics=[self.dice_coefficient, self.dice_coefficient_source, self.dice_coefficient_target])
        else:
            self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.dice_coefficient_loss, metrics=[self.dice_coefficient])

    def train_model(self, n_iteration):
        '''
        Compute all the patch_indexes, load in the memory the corresponding patches.
        For each of the n_iteration (epochs):
            - Compute the callbacks
            - Multiply the value of alpha by alpha-factor (not used in practice)
            While all the training set was not used for training:
                - Select a batch of source and target patches that were not seen before and load them
                - Get the predictions (deep-layer activation + output) of the network with the selected batch
                - Compute and change gamma with the predictions
                 - Train the network
            End While
            While all the validation set was not used:
                - Select random validation batches
                - Test the network with this validation batch
        :param n_iteration:
        :return:
        '''

        len_history = len(self.model.metrics_names) # Print self.model.metric_names to know the correspondance
        hist_l = np.empty((0,len_history))
        val_l = np.empty((0, len_history))
        epoch_hist = np.empty((0, len_history))
        epoch_val = np.empty((0, len_history))
        self.get_patch_indexes()
        if self.config.load_all_data:
            self.load_all_data(copy(self.complete_source_training_list), copy(self.complete_target_training_list),
                               copy(self.complete_source_validation_list), copy(self.complete_target_validation_list))
        count = 0
        for i in range(n_iteration):
            start_epoch = time.time()

            if self.config.callback:
                early_stop = self.callback(val_l)
                if early_stop:
                    break

            print("=============")
            print("Epoch:", i+1, "/", n_iteration)



            if i%20 == 0 and i !=0:
                #Increasing alpha every 20 epochs
                K.set_value(self.jdot_alpha, K.get_value(self.jdot_alpha)*self.config.alpha_factor)
                print("Changing jdot's alpha to :", K.get_value(self.jdot_alpha))

            while not self.epoch_complete:
                selected_source, selected_target = self.select_indices_training()
                if len(selected_source) < self.batch_size or len(selected_target) < self.batch_size:
                    break
                if self.config.load_all_data:
                    self.load_training_batch_from_all_data(selected_source, selected_target)
                else:
                    self.load_batch(selected_source, selected_target)

                intermediate_output = [self.get_prediction()] if not self.config.depth_jdot else self.get_prediction()
                self.prediction = intermediate_output[-1] #The output segmentation map

                K.set_value(self.target_pred, self.prediction[self.batch_size:,:])
                K.set_value(self.source_truth, self.train_batch[1][:self.batch_size, :])

                K.set_value(self.gamma, self.compute_gamma(self.prediction))
                epoch_hist = self.train_on_batch(epoch_hist)

            while not self.validation_complete:
                selected_source, selected_target = self.select_indices_validation()
                if self.config.load_all_data:
                    self.load_validation_batch_from_all_data(selected_source, selected_target)
                else:
                    self.load_validation_batch(selected_source, selected_target)
                epoch_val = self.test_on_batch(epoch_val)

            end_epoch = time.time()
            time_epoch = end_epoch - start_epoch
            epoch_remaining = n_iteration - (i+1)

            mean_epoch = np.mean(epoch_hist, axis=0)
            mean_val = np.mean(epoch_val, axis=0)


            self.pretty_print(mean_epoch, mean_val, time_epoch, epoch_remaining)
            hist_l = np.vstack((hist_l, [mean_epoch]))
            val_l = np.vstack((val_l, [mean_val]))


            self.epoch_complete = False
            self.validation_complete = False

            self.save_hist_and_model(hist_l, val_l)


    def train_model_on_source(self, n_iteration):
        '''
        Same idea as train_model but we don't need to compute gamma.
        :param n_iteration:
        :return:
        '''
        len_history = len(self.model.metrics_names)  # Print self.model.metric_names to know the correspondance
        hist_l = np.empty((0, len_history))
        val_l = np.empty((0, len_history))
        epoch_hist = np.empty((0, len_history))
        epoch_val = np.empty((0, len_history))
        self.get_patch_indexes()
        if self.config.load_all_data:
            self.load_all_data(copy(self.complete_source_training_list), copy(self.complete_target_training_list),
                               copy(self.complete_source_validation_list), copy(self.complete_target_validation_list), target=False)

        for i in range(n_iteration):
            start_epoch = time.time()

            if self.config.callback:
                early_stop = self.callback(val_l)
                if early_stop:
                    break

            print("=============")
            print("Epoch:", i + 1, "/", n_iteration)

            while not self.epoch_complete:
                selected_source, selected_target = self.select_indices_training()
                if len(selected_source) < self.batch_size or len(selected_target) < self.batch_size:
                    break
                if self.config.load_all_data:
                    self.load_training_batch_from_all_data(selected_source, selected_target, target=False)
                else:
                    self.load_batch(selected_source, selected_target, target=False)
                intermediate_output = [self.get_prediction()] if not self.config.depth_jdot else self.get_prediction()
                self.prediction = intermediate_output[-1]  # The output segmentation map

                epoch_hist = self.train_on_batch(epoch_hist)

            while not self.validation_complete:
                selected_source, selected_target = self.select_indices_validation()
                if self.config.load_all_data:
                    self.load_validation_batch_from_all_data(selected_source, selected_target, target=False)
                else:
                    self.load_validation_batch(selected_source, selected_target, target=False)
                epoch_val = self.test_on_batch(epoch_val)

            end_epoch = time.time()
            time_epoch = end_epoch - start_epoch
            epoch_remaining = n_iteration - (i + 1)

            mean_epoch = np.mean(epoch_hist, axis=0)
            mean_val = np.mean(epoch_val, axis=0)

            self.pretty_print(mean_epoch, mean_val, time_epoch, epoch_remaining)
            hist_l = np.vstack((hist_l, [mean_epoch]))
            val_l = np.vstack((val_l, [mean_val]))

            self.epoch_complete = False
            self.validation_complete = False

            if val_l.size != 0 and val_l[0][-1] <= np.all(val_l[0]):
                # We save the model if it's the best one existing.
                self.save_hist_and_model(hist_l, val_l)

    def callback(self, val_l, min_delta_lr = 0.0001, min_delta_stop = 0.00001):
        '''
        Two callbacks:
            - Reduce lr on plateau
            - Early stopping
        They both mimic the callbacks implemented in keras.
        :param val_l:
        :param min_delta_lr: threshold of significant changes for reduce lr on plateau
        :param min_delta_stop: threshold of significant changes for early stop
        :return:
        '''
        if val_l.shape[0] > self.config.patience + 2 and all(
                [val_l[-(x+2)][0] <= val_l[-(x+1)][0]+min_delta_lr*val_l[-(x+1)][0] for x in range(self.config.patience)]) and self.count == 0:
            # We let #patience# epochs run before starting to monitor the loss
            # We monitor the loss and multiply the learnig rate by #learning rate drop# if it didn't improve for #patience# epochs
            K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * self.config.learning_rate_drop)
            print("Reducing learning rate on plateau: ", K.get_value(self.model.optimizer.lr))
            self.count = self.config.patience
        if self.count > 0:
            self.count = self.count - 1

        if val_l.shape[0] > self.config.early_stop + 2 and all(
                [val_l[-(x+2)][0] <= val_l[-(x+1)][0]+min_delta_stop*val_l[-(x+1)][0] for x in range(self.config.early_stop)]):
            # We let #early stop# epochs run before starting to monitor the loss
            print("Early stopping")
            return True
        else:
            return False

    def get_batch(self, selected_source, selected_target, target = True, validation=False, all = False):
        """
        Function to get a random batch of source and target images
        :return: Two tuples (image samples,ground_truth). From 0 to the batch_size the image samples belong to the
        source domain from the batch_size until the end image samples belong to the target domain.
        The image samples are of shape (number_of_modalities, patch_shape)
        """

        batch, affine_list_source, affine_list_target = get_batch_jdot(
            selected_source, selected_target,
            self.source_data,
            self.target_data,
            batch_size=self.config.batch_size,
            data_split=self.config.validation_split,
            overwrite_data=self.config.overwrite_data,
            validation_keys_file=self.config.validation_file,
            training_keys_file=self.config.training_file,
            n_labels=self.config.n_labels,
            labels=self.config.labels,
            patch_shape=self.config.patch_shape,
            validation_batch_size=self.config.validation_batch_size,
            validation_patch_overlap=self.config.validation_patch_overlap,
            training_patch_overlap=self.config.training_patch_overlap,
            training_patch_start_offset=self.config.training_patch_start_offset,
            permute=self.config.permute,
            augment=self.config.augment,
            skip_blank=self.config.skip_blank,
            augment_flip=self.config.flip,
            augment_distortion_factor=self.config.distort,
            number_of_threads= self.config.number_of_threads,
            target=target,
            validation = validation,
            source_center = self.config.source_center,
            target_center = self.config.target_center,
            all = all)
        return batch, affine_list_source, affine_list_target

    def get_patch_indexes(self, target = True):
        '''
        Compute the list of patches we wan't to use for the training.
        :param target:
        :return:
        '''
        self.complete_source_training_list, self.complete_source_validation_list, self.complete_target_training_list, \
        self.complete_target_validation_list = get_patches_index_list(
                               self.source_data, self.target_data,
                               training_keys_file_source=self.config.training_file_source,
                               validation_keys_file_source=self.config.validation_file_source,
                               training_keys_file_target=self.config.training_file_target,
                               validation_keys_file_target=self.config.validation_file_target,
                               source_center=self.config.source_center,
                               target_center=self.config.target_center,
                               data_split=self.config.validation_split,
                               change_validation=self.config.change_validation,
                               patch_shape=self.config.patch_shape,
                               skip_blank=self.config.skip_blank,
                               training_patch_overlap=self.config.training_patch_overlap,
                               validation_patch_overlap=self.config.validation_patch_overlap,
                               training_patch_start_offset=self.config.training_patch_start_offset,
                               split_list = self.config.split_list)

        self.source_training_list = copy(self.complete_source_training_list)
        self.source_validation_list = copy(self.complete_source_validation_list)
        self.target_training_list = copy(self.complete_target_training_list)
        self.target_validation_list = copy(self.complete_target_validation_list)
        print("Source training: ", len(self.complete_source_training_list))
        print("Source validation", len(self.complete_source_validation_list))
        print("Target training", len(self.complete_target_training_list))
        print("Target validation", len(self.complete_target_validation_list))

    def select_indices_training(self):
        '''
        Shuffle the source and target training list
        Pop #batch_size# samples.
        If there is not enough patches in source or target training list mark the epoch as complete and reset both of them with the complete lists
        :return: The selected source and target samples
        '''
        random.shuffle(self.source_training_list)
        random.shuffle(self.target_training_list)
        selected_source = []
        selected_target = []
        while len(selected_source) < self.batch_size and len(self.source_training_list) > 0 and len(self.target_training_list) > 0:
            selected_source += [self.source_training_list.pop()]
            selected_target += [self.target_training_list.pop()]

        if len(self.source_training_list) < self.batch_size or len(self.target_training_list) < self.batch_size:
            self.source_training_list = copy(self.complete_source_training_list)
            self.target_training_list = copy(self.complete_target_training_list)
            self.epoch_complete = True

        return selected_source, selected_target
    
    def select_indices_validation(self):
        '''
        Same as before for validation.
        :return:
        '''
        random.shuffle(self.source_validation_list)
        random.shuffle(self.target_validation_list)
        selected_source = []
        selected_target = []
        while len(selected_source) < self.batch_size and len(self.source_validation_list) > 0 and len(self.target_validation_list) > 0:
            selected_source += [self.source_validation_list.pop()]
            selected_target += [self.target_validation_list.pop()]
        if len(self.source_validation_list) < self.batch_size or len(self.target_validation_list) < self.batch_size:
            self.source_validation_list = copy(self.complete_source_validation_list)
            self.target_validation_list = copy(self.complete_target_validation_list)
            self.validation_complete = True
            
        return selected_source, selected_target

    def load_batch(self, selected_source, selected_target, target = True):
        '''
        Load the batch of patches from their indices when self.config.load_all_data == False.
        :param selected_source: List of selected patches for the source center
        :param selected_target: List of selected patches for the target center
        :param target:
        :return:
        '''
        start = time.time()
        self.train_batch, _, _ = self.get_batch(selected_source, selected_target, target=target)
        end = time.time()
        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()
        if target:
            K.set_value(self.batch_source, self.train_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.train_batch[0][self.batch_size:])


    def load_validation_batch(self, selected_source, selected_target, target = True):
        start = time.time()
        self.val_batch, _, _ = self.get_batch(selected_source, selected_target, target=target)
        end = time.time()
        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()
        if target:
            K.set_value(self.batch_source, self.val_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.val_batch[0][self.batch_size:])


    def load_training_batch_from_all_data(self, selected_source, selected_target, target = True):
        '''
        Function to load the batch from the data loaded in memory.
        We iterate through all the patches in self.complete_source_training_list and fetch the indices of the selected patches.
        Based on these indices, we get the corresponding image patches from the previously loaded patches.
        :param selected_source:
        :param selected_target:
        :param target:
        :return:
        '''
        index_source = []
        index_target = []
        start = time.time()
        for i, index in enumerate(copy(self.complete_source_training_list)):
            for j in selected_source:
                if index[0] == j[0] and (index[1] == j[1]).all():
                    index_source += [i]
        if target:
            for i, index in enumerate(copy(self.complete_target_training_list)):
                for j in selected_target:
                    if index[0] == j[0] and (index[1] == j[1]).all():
                        index_target += [i]
        selected_index = index_source + [i + len(self.complete_source_training_list) for i in index_target]
        self.train_batch = (np.array([self.training_data[0][i] for i in selected_index]), np.array([self.training_data[1][i] for i in selected_index]))
        affine_list = [self.affine_source_training[i] for i in index_source]
        affine_list += [self.affine_target_training[i] for i in index_target]
        if target:
            index_list = selected_source + selected_target
        else:
            index_list = selected_source
        if self.config.augment:
            self.train_batch = multi_proc_augment_data(self.train_batch, affine_list, index_list, patch_shape= self.config.patch_shape, augment=self.config.augment,
                                    augment_flip=self.config.flip, augment_distortion_factor=self.config.distort, skip_blank=self.config.skip_blank,
                                    permute=self.config.permute)
        if target:
            K.set_value(self.batch_source, self.train_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.train_batch[0][self.batch_size:])

        end = time.time()

        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()

    def load_validation_batch_from_all_data(self, selected_source, selected_target, target = True):
        '''
        Same as before for the validation data.
        :param selected_source:
        :param selected_target:
        :param target:
        :return:
        '''
        index_source = []
        index_target = []
        start = time.time()
        for i, index in enumerate(copy(self.complete_source_validation_list)):
            for j in selected_source:
                if index[0] == j[0] and (index[1] == j[1]).all():
                    index_source += [i]
        if target:

            for i, index in enumerate(copy(self.complete_target_validation_list)):
                for j in selected_target:
                    if index[0] == j[0] and (index[1] == j[1]).all():
                        index_target += [i]
        selected_index = index_source + [i + len(self.complete_source_validation_list) for i in index_target]
        self.val_batch = (np.array([self.validation_data[0][i] for i in selected_index]), np.array([self.validation_data[1][i] for i in selected_index]))
        affine_list = [self.affine_source_validation[i] for i in index_source]
        affine_list += [self.affine_target_validation[i] for i in index_target]
        if target:
            index_list = selected_source + selected_target
        else:
            index_list = selected_source
        if self.config.augment:
            self.val_batch = multi_proc_augment_data(self.validation_batch, affine_list, index_list, patch_shape= self.config.patch_shape, augment=self.config.augment,
                                    augment_flip=self.config.flip, augment_distortion_factor=self.config.distort, skip_blank=self.config.skip_blank,
                                    permute=self.config.permute)

        if target:
            K.set_value(self.batch_source, self.val_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.val_batch[0][self.batch_size:])

        end = time.time()

        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()


    def load_all_data(self, training_source, training_target, validation_source, validation_target, target = True):
        '''
        Load all the data in the memory
        :param training_source:
        :param training_target:
        :param validation_source:
        :param validation_target:
        :param target:
        :return:
        '''
        start = time.time()
        print("Loading training data: \n")
        self.training_data, self.affine_source_training, self.affine_target_training = self.get_batch(training_source, training_target, target=target, all = True)
        print("Loading validation data: \n")
        self.validation_data, self.affine_source_validation, self.affine_target_validation = self.get_batch(validation_source, validation_target, target=target, all =True)
        print("Training data: ", len(self.training_data[0]))
        print("Validation data: ", len(self.validation_data[0]))
        end = time.time()

        hour, minute, seconds = self.compute_time(end - start)
        print("Time for evaluation: ", hour, "hour(s)", minute, "minute(s)", seconds, "second(s)")

    def get_prediction(self):
        '''
        Function to get the prediction of the model at a step t.
        We fetch the representation at the layer we choosed in self.config.depth_jdot.

        :return:
        '''

        intermediate_output = self.model.predict(self.train_batch[0])

        if self.config.depth_jdot == None:
            self.image_representation_source = self.train_batch[0][:self.batch_size, :]
            self.image_representation_target = self.train_batch[0][self.batch_size:, :]
        else:
            # We fetch the first output which is the intermediate representation we are interested in
            self.image_representation_source = intermediate_output[0][:self.batch_size, :]
            self.image_representation_target = intermediate_output[0][self.batch_size:, :]


        return intermediate_output

    def train_on_batch(self, hist_l):
        '''
        Function to train the model on the loaded batch.
        We first start by creating dummies output for each deep layer we are interested in.
        We train the model ont the batch and fetch the history.
        :param validation: If true, a validation step is performed.
        :param hist_l: List of all the history step
        :param val_l:
        :return: Updated hist_l
        '''
        output_list = []
        for name in self.context_output_name: #Creating a bunch of false outputs
            output_list += [np.zeros((self.train_batch[0].shape[0],) + self.model.get_layer(name).output_shape[1:])]
        training_output_list = output_list + [self.train_batch[1]]

        # We train the model
        if self.config.train_jdot:
            hist = self.model.train_on_batch(self.train_batch[0], training_output_list)
        else:
            hist = self.model.train_on_batch(self.train_batch[0], self.train_batch[1])
        hist_l = np.vstack((hist_l, hist))
        if self.config.train_jdot:
            result = "\rLoss: " + str(hist[0]) + " Dice Score: " + str(hist[-3]) + " Dice Score Source: " + str(hist[-2]) + " Dice Score Target: " + str(hist[-1])
        else:
            result = "\rLoss: " + str(hist[0]) + " Dice Score: " + str(hist[-1])
        sys.stdout.write(result)
        sys.stdout.flush()

        return hist_l
    
    def test_on_batch(self, val_l):
        '''
        Function to test the network on the loaded batch.
        :param val_l:
        :return:
        '''
        output_list = []
        for name in self.context_output_name: #Creating a bunch of false outputs
            output_list += [np.zeros((self.val_batch[0].shape[0],) + self.model.get_layer(name).output_shape[1:])]
        validation_output_list = output_list + [self.val_batch[1]]

        # We val the model
        if self.config.train_jdot:
            hist = self.model.test_on_batch(self.val_batch[0], validation_output_list)
            result = "\r Validation Loss:" + str(hist[0]) + " Validation Dice Score: " + str(
                hist[-3]) + " Validation Dice Score Source: " + str(hist[-2]) + " Validation Dice Score Target: " + str(
                hist[-1])

        else:
            hist = self.model.test_on_batch(self.val_batch[0], self.val_batch[1])
            result = "\r Validation Loss:" + str(hist[0]) + " Validation Dice Score: " + str(
                hist[-1])

        val_l = np.vstack((val_l, hist))

        sys.stdout.write(result)
        sys.stdout.flush()

        return val_l

    def pretty_print(self, mean_epoch, mean_val, time_epoch, epoch_remaining):
        '''
        Function to print the results on an iteration and expected time of arrival.
        :param mean_epoch:
        :param mean_val:
        :param time_epoch:
        :param epoch_remaining:
        :return:
        '''
        delta = time_epoch*epoch_remaining
        hour, minute, seconds = self.compute_time(delta)
        if self.config.train_jdot:
            print("\n\nMean on epoch :")
            result = "Loss: " + str(mean_epoch[0]) + " Dice Score: " + str(mean_epoch[-3]) + " Dice Score Source: " + str(
                mean_epoch[-2]) + " Dice Score Target: " + str(mean_epoch[-1])
            print(result)
            print("\nMean on validation")
            result = "Loss: " + str(mean_val[0]) + " Dice Score: " + str(mean_val[-3]) + " Dice Score Source: " + str(
                mean_val[-2]) + " Dice Score Target: " + str(mean_val[-1])
            print(result)
            print("\nEstimated time remaining for training: ", hour, " hour(s) ", minute, " minute(s) ", int(seconds), " second(s) ")
            print("==============\n")

        else:
            print("\n\nMean on epoch :")
            result = "Loss: " + str(mean_epoch[0]) + " Dice Score: " + str(mean_epoch[-1])
            print(result)
            print("\nMean on validation")
            result = "Loss: " + str(mean_val[0]) + " Dice Score: " + str(mean_val[-1])
            print(result)
            print("\nEstimated time remaining for training: ", hour, " hour(s) ", minute, " minute(s) ", int(seconds), " second(s) ")
            print("==============\n")

    def compute_time(self, time):
        hour = int(time / 3600)
        time -= hour * 3600
        minute = int(time / 60)
        time -= minute * 60
        seconds = time
        return hour, minute, seconds


    def save_hist_and_model(self, hist_l, val_l):
        '''
        Function to save the model and the history
        :param hist_l:
        :param val_l:
        :return:
        '''
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        np.savetxt(os.path.join(self.config.save_dir, "validation.csv"), val_l, delimiter=",")
        np.savetxt(os.path.join(self.config.save_dir, "train.csv"), hist_l, delimiter=",")

        self.model.save(self.config.model_file)




    def compute_gamma(self, pred):
        '''
        Function to compute the OT between the target and source samples.
        :return:Gamma the OT matrix
        '''
        # Reshaping the samples into vectors of dimensions number of modalities * patch_dimension.
        # train_vecs are of shape (batch_size, d)
        train_vec_source = np.reshape(self.image_representation_source, (self.batch_size, self.image_representation_source.shape[1]*
                                                                         self.image_representation_source.shape[2]*
                                                                         self.image_representation_source.shape[3]*
                                                                         self.image_representation_source.shape[4]))
        
        train_vec_target = np.reshape(self.image_representation_target, (self.batch_size, self.image_representation_target.shape[1]*
                                                                         self.image_representation_target.shape[2]*
                                                                         self.image_representation_target.shape[3]*
                                                                         self.image_representation_target.shape[4]))
        # Same for the ground truth but the GT is the same for both modalities

        truth_vec_source = np.reshape(self.train_batch[1][:self.batch_size],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))
        pred_vec_source = np.reshape(pred[:self.batch_size],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))

        # We don't have information on target labels
        pred_vec_target = np.reshape(pred[self.batch_size:],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))

        # Compute the distance between samples and between the source_truth and the target prediction.
        C0 = cdist(train_vec_source, train_vec_target, metric="sqeuclidean")
        C1 = cdist(truth_vec_source, pred_vec_target, metric=self.config.jdot_distance)
        C = K.get_value(self.jdot_alpha)*C0+K.get_value(self.jdot_beta)*C1

        # Computing gamma using the OT library
        gamma = ot.emd(ot.unif(self.batch_size), ot.unif(self.batch_size), C)
        return gamma

    def evaluate_model(self):
        """
        Function to evaluate the trained model.
        We first begin by loading the old model and compile it (useful if you wan't to evaluate an old model).
        We then create the test dataset
        Finally we run the validation cases
        :return:
        """
        start = time.time()
        self.load_old_model(self.config.model_file)
        self.config.depth_jdot = None
        self.compile_model()
        self.context_output_name = None


        test = create_test.Test(self.config)
        test.main(overwrite_data=self.config.overwrite_data)

        self.run_validation_cases(validation_keys_file=self.config.validation_file,
                                  training_modalities=self.config.training_modalities,
                                  labels=self.config.labels,
                                  hdf5_file=self.config.data_file,
                                  output_label_map=True,
                                  overlap=self.config.validation_patch_overlap,
                                  output_dir=self.config.prediction_dir,
                                  save_image=self.config.save_image)
        end = time.time()
        hour, minute, seconds = self.compute_time(end - start)
        print("Time for evaluation: ", hour, "hour(s)", minute, "minute(s)", seconds, "second(s)")

    def run_validation_cases(self, validation_keys_file, training_modalities, labels, hdf5_file,
                             output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False,
                             save_image = False):
        '''
        For each patient of the testing set we run a validation.
        :param validation_keys_file:
        :param training_modalities:
        :param labels:
        :param hdf5_file:
        :param output_label_map:
        :param output_dir:
        :param threshold:
        :param overlap:
        :param permute:
        :param save_image:
        :return:
        '''
        validation_indices = pickle_load(validation_keys_file)
        model = self.model
        data_file = tables.open_file(hdf5_file, "r")
        for i, index in enumerate(validation_indices):
            actual = round(i/len(validation_indices)*100, 2)
            print("Running validation case: ", actual,"%")
            if 'subject_ids' in data_file.root:
                case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
            else:
                case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
            run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                                training_modalities=training_modalities, output_label_map=output_label_map, labels=labels,
                                threshold=threshold, overlap=overlap, permute=permute, save_image=save_image)
        data_file.close()

    def load_old_model(self, model_file):
        '''
        Function to load old model with our custom loss functions.
        :param model_file:
        :return:
        '''
        print("Loading pre-trained model")
        custom_objects = {'jdot_loss': self.jdot_image_loss,
                          'jdot_image_loss': self.jdot_image_loss,
                          'deep_jdot_loss': self.deep_jdot_loss,
                          'deep_jdot_loss_dice': self.deep_jdot_loss_dice,
                          'deep_jdot_loss_euclidean': self.deep_jdot_loss_euclidean,
                          'distance_loss': self.distance_loss,
                          'dice_coefficient': self.dice_coefficient,
                          'dice_coefficient_loss': self.dice_coefficient_loss,
                          'dice_coefficient_source': self.dice_coefficient_source,
                          'dice_coefficient_target': self.dice_coefficient_target}
        try:
            from keras_contrib.layers import InstanceNormalization
            custom_objects["InstanceNormalization"] = InstanceNormalization
        except ImportError:
            pass
        try:
            self.model = load_model(model_file, custom_objects=custom_objects)
        except ValueError as error:
            if 'InstanceNormalization' in str(error):
                raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                              "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
            else:
                raise error
