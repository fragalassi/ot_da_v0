from training_testing import create_test
import os
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_data_from_file, get_validation_split, create_patch_index_list, add_data

class Compare_patches:

    def __init__(self, conf):
        self.config = conf
        create = create_test.Test(self.config)
        self.fetch_testing_data_files = create.fetch_testing_data_files
        self.patch_shape = (64, 64, 32)


    def main(self):
        index_list, validation_list, data_file = self.get_index_list()

        for i in validation_list:
            list_a = [tup for tup in index_list if tup[0] == i]
            x_a, y_a = get_data_from_file(data_file, list_a, patch_shape=self.patch_shape)
            for j in validation_list:
                if i != j:
                    list_b = [tup for tup in index_list if tup[0] == j]
                    x_b, y_b = get_data_from_file(data_file, list_b, patch_shape=self.patch_shape)

                    self.compare_patches(x_a,y_a,x_b,y_b)




    def get_index_list(self, overwrite_data=True, patch_overlap=0, patch_start_offset = None):
        '''
        Function to get the indexes of all the images and all the patches we want to compare
        :param overwrite_data: If False it will read previously written files
        :param patch_overlap:
        :param patch_start_offset:
        :return:
        '''
        self.config.validation_split = 0.0
        self.config.data_file = os.path.abspath("Data/generated_data/" + self.config.data_set + "_testing.h5")
        self.config.training_file = os.path.abspath(
            "Data/generated_data/" + self.config.data_set + "_testing.pkl")
        self.config.validation_file = os.path.abspath(
            "Data/generated_data/" + self.config.data_set + "_testing_validation_ids.pkl")
        # convert input images into an hdf5 file
        if overwrite_data or not os.path.exists(self.config.data_file):
            testing_files, subject_ids = self.fetch_testing_data_files(return_subject_ids=True)
            write_data_to_file(testing_files, self.config.data_file, image_shape=self.config.image_shape,
                               subject_ids=subject_ids)
        data_file_opened = open_data_file(self.config.data_file)

        training_list, validation_list = get_validation_split(data_file_opened,
                                                              data_split=0,
                                                              overwrite_data=overwrite_data,
                                                              training_file=self.config.training_file,
                                                              validation_file=self.config.validation_file)

        index_list = create_patch_index_list(validation_list, data_file_opened.root.data.shape[-3:], self.patch_shape,
                                             patch_overlap, patch_start_offset)

        return index_list, validation_list, data_file_opened

    def compare_patches(self, x_a, y_a, x_b, y_b):
        print(x_a)


compare = Compare_patches