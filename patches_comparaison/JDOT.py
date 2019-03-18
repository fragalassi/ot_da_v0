from keras import backend as K
from patches_comparaison.generator_jdot import get_training_and_validation_batch_jdot
from scipy.spatial.distance import cdist, cosine
from unet3d.utils import pickle_load
import tables
from unet3d.prediction import run_validation_case
import os
import numpy as np
import ot
from training_testing import create_test
import time

class JDOT():

    def __init__(self, model, config, source_data, target_data, allign_loss=1.0, tar_cl_loss=1.0,
                 sloss=0.0, tloss=1.0, int_lr=0.01, ot_method='emd',
                 jdot_alpha=0.01, lr_decay=True, verbose=1):


        self.config = config
        self.model = model  # target model

        self.source_data = source_data
        self.target_data = target_data

        self.batch_size = self.config.batch_size
        self.optimizer = self.config.optimizer
        # initialize the gamma (coupling in OT) with zeros
        self.gamma = K.zeros(shape=(self.batch_size, self.batch_size))
        self.batch_source = K.zeros(shape=(self.batch_size,
                                    len(self.config.training_modalities),self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))
        self.batch_target = K.zeros(shape=(self.batch_size,
                                    len(self.config.training_modalities),self.config.patch_shape[0],self.config.patch_shape[1],self.config.patch_shape[2]))

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
        self.jdot_alpha = jdot_alpha  # weight for the alpha term

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
            Compute the cosine distance between each of the source samples and each of the target samples.
            It returns a matrix (batch_size, batch_size).
            This cosine distance is computed both in the image space and in the truth/prediction space.
            '''
            cos_distance_samples = cos_distance(K.batch_flatten(self.batch_source),K.batch_flatten(self.batch_target))
            cos_distance_pred = cos_distance(K.batch_flatten(truth_source), K.batch_flatten(prediction_target))

            return source_loss + K.sum(self.gamma*(cos_distance_samples + cos_distance_pred))

        self.jdot_loss = jdot_loss

        def dice_coefficient(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        self.dice_coefficient = dice_coefficient

        def dice_coefficient_loss(y_true, y_pred):
            return -dice_coefficient(y_true, y_pred)

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

        '''
        Uncomment to check if cos_distance is computing the right values.
        '''
        # x = np.array([[2, 5, 10], [64, 47, 10], [45, 35, 47]])
        # y = np.array([[2, 5,10], [4, 22, 156], [547, 48, 7]])
        # print(cosine(x[0],y[0]))
        # print(cosine(x[0], y[1]))
        # print(cosine(x[1],y[0]))
        # print("Evaluation: ", K.eval(cos_distance(K.constant(x), K.constant(y))))


    def compile_model(self):
        '''
        Compilation with the custom loss function and the metrics.
        :return:
        '''
        self.model.compile(optimizer=self.optimizer(lr=self.config.initial_learning_rate), loss=self.jdot_loss, metrics=[self.dice_coefficient])


    def get_batch(self):
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
            number_of_threads= self.config.number_of_threads)

        return train_batch, validation_batch


    def train_model(self, n_iteration):
        '''
        For every iteration we first load a batch and compute a prediction on this batch.
        From this prediction we compute the OT gamma.
        We then train the newtork given the OT and the batch.
        :param n_iteration:
        :return:
        '''
        hist_l = np.empty((0,2))
        val_l = np.empty((0, 2))
        for i in range(n_iteration):
            print("Epoch:", i, "/", n_iteration)
            start = time.time()
            self.train_batch, self.validation_batch = self.get_batch()
            end = time.time()
            print("Time for loading: ",end - start)
            K.set_value(self.batch_source, self.train_batch[0][:self.batch_size])
            K.set_value(self.batch_target, self.train_batch[0][self.batch_size:])

            pred = self.model.predict(self.train_batch[0])

            K.set_value(self.gamma, self.compute_gamma(pred))

            hist = self.model.train_on_batch(self.train_batch[0], self.train_batch[1])

            hist_l = np.vstack((hist_l, hist))
            ponderation = [1/2, 1/4, 1/8, 1/16, 1/16]
            average = np.average(hist_l[-10:], axis=0)

            print("Loss:", hist[0], " Dice Score: ", hist[1],"| Loss mean: ", average[0], "Dice Score mean:", average[1], "\n")

            if i % 10 == 0:
                val = self.model.test_on_batch(self.validation_batch[0], self.validation_batch[1])
                val_l = np.vstack((val_l, val))
                print("======")
                print("Validation Loss: ", val[0], "Dice Score :", val[1])
                print("======", "\n")

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

        train_vec_source = np.reshape(self.train_batch[0][:self.batch_size],
                                      (self.batch_size, len(self.config.training_modalities)*self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))
        train_vec_target = np.reshape(self.train_batch[0][self.batch_size:],
                                      (self.batch_size, len(self.config.training_modalities)*self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))
        # Same for the ground truth but the GT is the same for both modalities

        truth_vec_source = np.reshape(self.train_batch[1][:self.batch_size],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))
        pred_vec_source = np.reshape(pred[:self.batch_size],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))

        # We don't have information on target labels
        pred_vec_target = np.reshape(pred[self.batch_size:],
                                      (self.batch_size, self.config.patch_shape[0]*self.config.patch_shape[1]*self.config.patch_shape[2]))

        # Compute the distance between samples and between the source_truth and the target prediction.
        C0 = cdist(train_vec_source, train_vec_target, metric="cosine")
        C1 = cdist(truth_vec_source, pred_vec_target, metric="cosine")

        # Resulting cost metric
        C = self.jdot_alpha*C0+K.eval(self.tloss)*C1

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
