from keras import backend as K
from keras import Model
from keras.callbacks import LambdaCallback
from patches_comparaison.generator_jdot import get_batch_jdot,\
    get_validation_split, create_patch_index_list, get_data_from_file, add_data_mp, get_patches_index_list
from scipy.spatial.distance import cdist, cosine, euclidean, dice
from unet3d.utils import pickle_load
import tables
from unet3d.prediction import run_validation_case
import os
import numpy as np
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

        self.t = 0

        self.prediction = []

        self.target_pred = K.zeros(shape=(self.batch_size, 1, self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))
        self.source_truth = K.zeros(shape=(self.batch_size, 1, self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))

        def deep_jdot_loss(y_true, y_pred):
            truth_source = y_true[:self.batch_size, :]  # source true labels
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :] # target prediction

            '''
            Compute the loss function of the source samples.
            '''
            source_loss = dice_coefficient_loss(truth_source, prediction_source)
            target_loss = euclidean_dist(K.batch_flatten(self.source_truth), K.batch_flatten(prediction_target))

            return source_loss + self.jdot_alpha * K.sum(self.gamma * target_loss)
        self.deep_jdot_loss = deep_jdot_loss

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

        def dice_coefficient(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient = dice_coefficient

        def dice_coefficient_target(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true[self.batch_size:, :])
            y_pred_f = K.flatten(y_pred[self.batch_size:, :])
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient_target = dice_coefficient_target

        def dice_coefficient_source(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true[:self.batch_size, :])
            y_pred_f = K.flatten(y_pred[:self.batch_size, :])
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient_source = dice_coefficient_source


        def dice_coefficient_loss(y_true, y_pred):
            return 1-dice_coefficient(y_true, y_pred)
        self.dice_coefficient_loss = dice_coefficient_loss
        '''
        Uncomment to check if cos_distance/euclidean_dist is computing the right values.
        '''
        # x = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
        # y = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 1]])
        # z = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        # print(K.eval(K.constant(x)-K.constant(y)))
        # print(K.eval(K.constant(z)*(K.constant(x) - K.constant(y))))
        # print(dice(x[0],y[0]))
        # print(dice(x[1], y[1]))
        # print(dice(x[2],y[2]))
        # print("Evaluation: \n", K.eval(dice_coefficient_loss(K.constant(x), K.constant(y))))

        def cos_distance(x,y):
            """
            Function to compute the cosine distance between two tensors.
            :param x:
            :param y:
            :return:
            """
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            return 1 - K.dot(x,K.transpose(y))

        def euclidean_dist(x,y):
            """
            The euclidean distance between x and y is the length of the displacement vector x - y:
            ||x-y||² = <x-y,x-y> (we take the square of the euclidean distance).
            <x-y,x-y> = <x,x> - <x,y> - <y,x> + <y,y>
            As both x and y lie in R, the dot product <.,.> is symmetric. Therefore: <x,y> = <y,x>.
            We can write:
            <x-y,x-y> = <x,x> - 2*<x,y> + <y,y>
            <x-y,x-y> = x'x - 2x'y + y'y
            :param x:
            :param y:
            :return:
            """
            dist = K.reshape(K.sum(K.square(x), 1), (-1, 1))
            dist += K.reshape(K.sum(K.square(y), 1), (1, -1))
            dist -= 2.0*K.dot(x, K.transpose(y))

            return K.sqrt(dist)

        def distance_loss(y_true, y_pred):
            prediction_source = y_pred[:self.batch_size, :]  # source prediction
            prediction_target = y_pred[self.batch_size:, :]  # target prediction
            dif = euclidean_dist(K.batch_flatten(prediction_source), K.batch_flatten(prediction_target))
            return self.jdot_alpha * K.sum(self.gamma*dif)

        self.distance_loss = distance_loss




        '''
        Uncomment to check if cos_distance/euclidean_dist is computing the right values.
        '''
        # x = np.array([[[[[10]]], [[[20]]], [[[0]]]], [[[[4]]], [[[7]]], [[[2]]]]])
        # y = np.array([[[[[2]]], [[[9]]], [[[3]]]], [[[[100]]], [[[10]]], [[[1]]]]])
        # # x = np.array([[1,9,3], [1,2,5]])
        # # y = np.array([[1,2,10], [1,4,3]])
        # print(euclidean(x[0],y[0]))
        # print(euclidean(x[0], y[1]))
        # print("Evaluation: \n", K.eval(euclidean_dist(K.constant(x), K.constant(y))))


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
                print("This model will be trained on the following losses :")
                print(loss)
                self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=loss, metrics=[self.dice_coefficient, self.dice_coefficient_source, self.dice_coefficient_target])
        else:
            self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.dice_coefficient_loss, metrics=[self.dice_coefficient])

    def train_model(self, n_iteration):
        '''
        For every iteration we first load a batch and compute a prediction on this batch.
        From this prediction we compute the OT gamma.
        We then train the newtork given the OT and the batch.
        :param n_iteration:
        :return:
        '''

        len_history = len(self.model.metrics_names) # Print self.model.metric_names to know the correspondance
        hist_l = np.empty((0,len_history))
        val_l = np.empty((0, len_history))
        epoch_hist = np.empty((0, len_history))
        epoch_val = np.empty((0, len_history))
        self.get_patch_indexes()

        for i in range(n_iteration):
            start_epoch = time.time()
            print("=============")
            print("Epoch:", i+1, "/", n_iteration)

            if i%20 == 0 and i !=0:
                #Increasing alpha every 10 epochs
                K.set_value(self.jdot_alpha, K.get_value(self.jdot_alpha)*self.config.alpha_factor)
                print("Changing jdot's alpha to :", K.get_value(self.jdot_alpha))

            while not self.epoch_complete:
                selected_source, selected_target = self.select_indices_training()
                if len(selected_source) < self.batch_size or len(selected_target) < self.batch_size:
                    break
                self.load_batch(selected_source, selected_target)
                intermediate_output = [self.get_prediction()] if not self.config.depth_jdot else self.get_prediction()
                self.prediction = intermediate_output[-1] #The output segmentation map

                K.set_value(self.target_pred, self.prediction[self.batch_size:,:])
                K.set_value(self.source_truth, self.train_batch[1][:self.batch_size, :])

                K.set_value(self.gamma, self.compute_gamma(self.prediction))
                epoch_hist = self.train_on_batch(epoch_hist)

            while not self.validation_complete:
                selected_source, selected_target = self.select_indices_validation()
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
        len_history = len(self.model.metrics_names)  # Print self.model.metric_names to know the correspondance
        hist_l = np.empty((0, len_history))
        val_l = np.empty((0, len_history))
        epoch_hist = np.empty((0, len_history))
        epoch_val = np.empty((0, len_history))
        self.get_patch_indexes()

        for i in range(n_iteration):
            start_epoch = time.time()
            print("=============")
            print("Epoch:", i + 1, "/", n_iteration)

            while not self.epoch_complete:
                selected_source, selected_target = self.select_indices_training()
                if len(selected_source) < self.batch_size or len(selected_target) < self.batch_size:
                    break
                self.load_batch(selected_source, selected_target, target=False)
                intermediate_output = [self.get_prediction()] if not self.config.depth_jdot else self.get_prediction()
                self.prediction = intermediate_output[-1]  # The output segmentation map

                epoch_hist = self.train_on_batch(epoch_hist)

            while not self.validation_complete:
                selected_source, selected_target = self.select_indices_validation()
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

            self.save_hist_and_model(hist_l, val_l)

    def get_batch(self, selected_source, selected_target, target = True, validation=False):
        """
        Function to get a random batch of source and target images
        :return: Two tuples (image samples,ground_truth). From 0 to the batch_size the image samples belong to the
        source domain from the batch_size until the end image samples belong to the target domain.
        The image samples are of shape (number_of_modalities, patch_shape)
        """
        batch = get_batch_jdot(
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
            target_center = self.config.target_center)
        return batch

    def get_patch_indexes(self, target = True):

        self.complete_source_training_list, self.complete_source_validation_list, self.complete_target_training_list, \
        self.complete_target_validation_list = get_patches_index_list(
                               self.source_data, self.target_data,
                               training_keys_file=self.config.training_file,
                               validation_keys_file=self.config.validation_file,
                               source_center=self.config.source_center,
                               target_center=self.config.target_center,
                               data_split=self.config.validation_split,
                               overwrite_data=self.config.change_validation,
                               patch_shape=self.config.patch_shape,
                               skip_blank=self.config.skip_blank,
                               training_patch_overlap=self.config.training_patch_overlap,
                               validation_patch_overlap=self.config.validation_patch_overlap,
                               training_patch_start_offset=self.config.training_patch_start_offset)

        self.source_training_list = copy(self.complete_source_training_list)
        self.source_validation_list = copy(self.complete_source_validation_list)
        self.target_training_list = copy(self.complete_target_training_list)
        self.target_validation_list = copy(self.complete_target_validation_list)
        print("Source training: ", len(self.complete_source_training_list))
        print("Source validation", len(self.complete_source_validation_list))
        print("Target training", len(self.complete_target_training_list))
        print("Target validation", len(self.complete_target_validation_list))
        if len(self.complete_source_validation_list) < self.batch_size:
            self.batch_size = len(self.complete_source_validation_list)
            print("Changing batch size to : ", len(self.complete_source_validation_list))
        elif len(self.complete_target_validation_list) < self.batch_size:
            self.batch_size = len(self.complete_target_validation_list)
            print("Changing batch size to : ", len(self.complete_target_validation_list))

    def select_indices_training(self):
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
        start = time.time()
        self.train_batch = self.get_batch(selected_source, selected_target, target=target)
        end = time.time()
        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()
        if target:
            K.set_value(self.batch_source, self.train_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.train_batch[0][self.batch_size:])

    def load_validation_batch(self, selected_source, selected_target, target = True):
        start = time.time()
        self.val_batch = self.get_batch(selected_source, selected_target, target=target)
        end = time.time()
        t = "\rTime for loading: " + str(end - start)
        sys.stdout.write(t)
        sys.stdout.flush()
        if target:
            K.set_value(self.batch_source, self.val_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.val_batch[0][self.batch_size:])

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
        :return:
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
        delta = time_epoch*epoch_remaining
        hour = int(delta / 3600)
        delta -= hour * 3600
        minute = int(delta / 60)
        delta -= minute * 60
        seconds = delta
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
        C1 = cdist(truth_vec_source, pred_vec_target, metric="sqeuclidean")

        # Resulting cost metric
        C = K.get_value(self.jdot_alpha)*(C0+C1)
        # Computing gamma using the OT library

        gamma = ot.emd(ot.unif(self.batch_size), ot.unif(self.batch_size), C)
        return gamma

    def evaluate_model(self):
        """
        Function to evaluate the trained model.
        :return:
        """
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
                                  output_dir=self.config.prediction_dir)


    def run_validation_cases(self, validation_keys_file, training_modalities, labels, hdf5_file,
                             output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False):
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
                                threshold=threshold, overlap=overlap, permute=permute)
        data_file.close()

    def load_old_model(self, model_file):
        print("Loading pre-trained model")
        custom_objects = {'jdot_loss': self.jdot_image_loss,
                          'jdot_image_loss': self.jdot_image_loss,
                          'deep_jdot_loss': self.deep_jdot_loss,
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
