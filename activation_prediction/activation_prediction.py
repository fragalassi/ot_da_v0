import keras
from unet3d.training import load_old_model
from unet3d.metrics import weighted_dice_coefficient_loss, dice_coef
from unet3d.generalized_loss import generalized_dice_loss, dice
#
class predict_activation:

    def __init__(self, conf):
        self.config = conf
        self.model = load_old_model(self.config.model_file)

    def main(self):
        loss_function_d = {
            "weighted_dice_coefficient_loss": weighted_dice_coefficient_loss,
            "generalized_dice_loss": generalized_dice_loss
        }
        self.model.compile(optimizer='Adam', loss=loss_function_d[self.config.loss_function])
        print(self.model.summary())