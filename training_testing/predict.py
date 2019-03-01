import os
#--------FG S
import sys 
# Add the folder path to the sys.path list
sys.path.append('/udd/aackaouy/OT-DA/')
#--------FG E

from unet3d.prediction import run_validation_cases

class Predict:
    def __init__(self, conf):
        self.config = conf

#config["model_file"] = os.path.abspath("miccai16_isensee_2017_model.h5")

    def main(self):
        prediction_dir = os.path.abspath("results/prediction/rev_"+str(self.config.rev)+"/prediction_"+self.config.data_set)
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        self.config.model_file = os.path.abspath("Data/generated_data/"+self.config.data_set+"_isensee_2017_model_rev"+str(self.config.rev)+".h5") #  patch (128,128,128) ; n_filters = 16; ski_blank = True; depth = 5

        self.config.data_file = os.path.abspath("Data/generated_data/" + self.config.data_set + "_testing.h5")
        self.config.validation_file = os.path.abspath("Data/generated_data/" + self.config.data_set + "_testing_validation_ids.pkl")

        run_validation_cases(validation_keys_file=self.config.validation_file,
                             model_file=self.config.model_file,
                             training_modalities=self.config.training_modalities,
                             labels=self.config.labels,
                             hdf5_file=self.config.data_file,
                             output_label_map=True,
                             overlap=self.config.validation_patch_overlap,
                             output_dir=prediction_dir)
