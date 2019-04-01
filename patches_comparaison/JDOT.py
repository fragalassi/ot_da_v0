from keras import backend as K
from keras import Model
from keras.callbacks import LambdaCallback
from patches_comparaison.generator_jdot import get_training_and_validation_batch_jdot, get_validation_split, create_patch_index_list, get_data_from_file, add_data_mp
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
from multiprocessing import Pool

class JDOT():

    def __init__(self, model, config, source_data, target_data, context_output_name, allign_loss=1.0, tar_cl_loss=1.0,
                 sloss=0.0, tloss=1.0, int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):


        self.config = config
        self.model = model  # target model
        self.context_output_name = context_output_name
        self.source_data = source_data
        self.target_data = target_data

        self.batch_size = self.config.batch_size
        self.optimizer = self.config.optimizer
        self.jdot_alpha = self.config.jdot_alpha
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


        def jdot_loss(y_true, y_pred):
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

        self.jdot_loss = jdot_loss

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
        self.dice_loss = dice_coefficient_loss
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
            dist = K.reshape(K.sum(K.square(x),1), (-1,1))
            dist += K.reshape(K.sum(K.square(y),1), (1,-1))
            dist -= 2.0*K.dot(x, K.transpose(y))

            return K.sqrt(dist)


        '''
        Uncomment to check if cos_distance/euclidean_dist is computing the right values.
        '''
        # x = np.array([[0, 0, 0], [64, 47, 10], [45, 35, 47]])
        # y = np.array([[0, 0, 0], [4, 22, 156], [547, 48, 7]])
        # print(euclidean(x[0],y[0]))
        # print(euclidean(x[0], y[1]))
        # print(euclidean(x[2],y[2]))
        # print("Evaluation: \n", K.eval(euclidean_dist(K.constant(x), K.constant(y))))


    def compile_model(self):
        '''
        Compilation with the custom loss function and the metrics.
        :return:
        '''
        if self.config.train_jdot:
            if self.config.depth_jdot == None:
                self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.jdot_loss, metrics=[self.dice_coefficient, self.dice_coefficient_source, self.dice_coefficient_target])
            else:
                outputs = [self.model.get_layer(name).output for name in self.context_output_name]
                outputs += [self.model.layers[-1].output]
                self.model = Model(inputs=self.model.input,
                                                 outputs=outputs)
                print(self.model.summary())
                self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.jdot_loss, metrics=[self.dice_coefficient, self.dice_coefficient_source, self.dice_coefficient_target])
        else:
            self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.dice_loss, metrics=[self.dice_coefficient])


    def get_batch(self, target = True, validation=False):
        """
        Function to get a random batch of source and target images
        :return: Two tuples (image samples,ground_truth). From 0 to the batch_size the image samples belong to the
        source domain from the batch_size until the end image samples belong to the target domain.
        The image samples are of shape (number_of_modalities, patch_shape)
        """
        train_batch, validation_batch = get_training_and_validation_batch_jdot(
            self.source_data, self.target_data,
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
        return train_batch, validation_batch

    def load_batch(self, validation):
        start = time.time()
        self.train_batch, self.validation_batch = self.get_batch(target=True, validation=validation)
        end = time.time()
        print("Time for loading: ", end - start)
        K.set_value(self.batch_source, self.train_batch[0][:self.batch_size])
        K.set_value(self.batch_target, self.train_batch[0][self.batch_size:])

    def get_prediction(self):

        outputs = [self.model.get_layer(name).output for name in self.context_output_name]
        outputs += [self.model.layers[-1].output]
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=outputs)
        intermediate_output = intermediate_layer_model.predict(self.train_batch[0])

        if self.config.depth_jdot == None:
            self.image_representation_source = self.train_batch[0][:self.batch_size, :]
            self.image_representation_target = self.train_batch[0][self.batch_size:, :]
        else:
            self.image_representation_source = intermediate_output[self.config.depth_jdot][:self.batch_size, :]
            self.image_representation_target = intermediate_output[self.config.depth_jdot][self.batch_size:, :]

        return intermediate_output

    def train_on_batch(self, validation, hist_l, val_l):
        output_list = []
        for name in self.context_output_name: #Creating a bunch of false outputs
            output_list += [np.zeros((self.train_batch[0].shape[0],) + self.model.get_layer(name).output_shape[1:])]
        output_list += [self.train_batch[1]]
        hist = self.model.train_on_batch(self.train_batch[0], output_list)
        print(hist)
        hist_l = np.vstack((hist_l, hist))

        print("Loss:", hist[0], " Dice Score: ", hist[1], "Dice Score Source: ", hist[2], "Dice Score Target: ",
              hist[3], "\n")

        if validation:
            val = self.model.test_on_batch(self.validation_batch[0], self.validation_batch[1])
            val_l = np.vstack((val_l, val))
            print("======")
            print("Validation Loss: ", val[0], "Dice Score :", val[1], "Dice Score Source: ", val[2],
                  "Dice Score Target: ", val[3], )
            print("======", "\n")

        self.save_hist_and_model(hist_l, val_l)

        return hist_l, val_l

    def save_hist_and_model(self, hist_l, val_l):
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        np.savetxt(os.path.join(self.config.save_dir, "validation.csv"), val_l, delimiter=",")
        np.savetxt(os.path.join(self.config.save_dir, "train.csv"), hist_l, delimiter=",")

        self.model.save(self.config.model_file)

    def train_model(self, n_iteration):
        '''
        For every iteration we first load a batch and compute a prediction on this batch.
        From this prediction we compute the OT gamma.
        We then train the newtork given the OT and the batch.
        :param n_iteration:
        :return:
        '''

        hist_l = np.empty((0,4))
        val_l = np.empty((0, 4))

        for i in range(n_iteration):
            print("Batch:", i, "/", n_iteration)
            if i % 10 == 0:
                validation = True
            else:
                validation = False

            if i%100 == 0 and i != 0:
                '''
                Increasing the weights of jdot every 100 epochs
                '''
                self.jdot_alpha *= 2
                print("Increasing JDOT alpha: ", self.jdot_alpha)

            self.load_batch(validation)
            intermediate_output = self.get_prediction()

            K.set_value(self.gamma, self.compute_gamma(intermediate_output[-1]))
            hist_l, val_l = self.train_on_batch(validation, hist_l, val_l)
            self.save_hist_and_model(hist_l, val_l)


    def train_model_on_source(self, n_iteration):
        hist_l = np.empty((0,2))
        val_l = np.empty((0, 2))
        for i in range(n_iteration):

            print("Epoch:", i, "/", n_iteration)
            if i % 10 == 0:
                validation = True
            else:
                validation = False
            start = time.time()
            self.train_batch, self.validation_batch = self.get_batch(target=False, validation=validation)
            end = time.time()
            print("Time for loading: ",end - start)
            hist = self.model.train_on_batch(self.train_batch[0], self.train_batch[1])

            hist_l = np.vstack((hist_l, hist))
            average = np.average(hist_l[-10:], axis=0)

            print("Loss:", hist[0], " Dice Score: ", hist[1], "| Loss mean: ", average[0], "Dice Score mean:",
                  average[1], "\n")

            if validation:
                val = self.model.test_on_batch(self.validation_batch[0], self.validation_batch[1])
                val_l = np.vstack((val_l, val))
                print("======")
                print("Validation Loss: ", val[0], "Dice Score :", val[1])
                print("======", "\n")


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
        C0 = cdist(train_vec_source, train_vec_target, metric="euclidean")
        C1 = cdist(truth_vec_source, pred_vec_target, metric="euclidean")

        # Resulting cost metric
        C = abs(C0-C1)

        # Computing gamma using the OT library

        gamma = ot.emd(ot.unif(self.batch_size), ot.unif(self.batch_size), C)
        return gamma

    def evaluate_model(self):
        """
        Function to evaluate the trained model.
        :return:
        """
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
        custom_objects = {'jdot_loss': self.jdot_loss,
                          'dice_coefficient': self.dice_coefficient,
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
