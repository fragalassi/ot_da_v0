import os
#--------FG S
import sys 
# Add the folder path to the sys.path list
sys.path.append('/temp_dd/igrida-fs1/fgalassi/3DUnetCNN-master/')
#--------FG E

from train import config
from unet3d.prediction import run_validation_cases

#config["model_file"] = os.path.abspath("miccai16_isensee_2017_model.h5")
config["training_modalities"] = ["T1-norm-include","FLAIR-norm-include"] #,"ATLAS-wm_mask-reg-include","ATLAS-gm_mask-reg-include","ATLAS-csf_mask-reg-include", "mask-int-include"]; 
 
def main():
    prediction_dir = os.path.abspath("prediction_miccai")

    config["model_file"] = os.path.abspath("miccai16_isensee_2017_model.h5") #  rev 2:patch (128,128,128) ; n_filters = 16; ski_blank = True; depth = 5
    config["model_file"] = os.path.abspath("miccai16_isensee_2017_model_rev2.h5") #  rev 2: patch (176,...) ; n_filters = 16; ski_blank = True; depth = 5; metric: Tze
    config["model_file"] = os.path.abspath("miccai16_isensee_2017_model_rev3.h5") #  patch (128,128,128) ; n_filters = 16; ski_blank = True; depth = 5

    config["data_file"] = os.path.abspath("testing_FG_rev0.h5")
    config["validation_file"] = os.path.abspath("testing_FG_ids_rev0.pkl")
    
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)

if __name__ == "__main__":
    main()
