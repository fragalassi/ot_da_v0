import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Evalaute:
    def __init__(self, conf):
        self.config = conf


    def get_whole_tumor_mask(self, data):
        return data > 0


    def get_tumor_core_mask(self, data):
        return np.logical_or(data == 1, data == 4)


    def get_enhancing_tumor_mask(self, data):
        return data == 4


    def dice_coefficient(self, truth, prediction):
        return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


    def main(self):
    #    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    #    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
        header = ["MSlesions"]
        masking_functions = (self.get_whole_tumor_mask)
        rows = list()
        subject_ids = list()
        for case_folder in glob.glob(os.path.join(os.path.dirname(__file__), "prediction","*")):
            if not os.path.isdir(case_folder):
                continue
            subject_ids.append(os.path.basename(case_folder))
            truth_file = os.path.join(case_folder, "truth.nii.gz")
            truth_image = nib.load(truth_file)
            truth = truth_image.get_data()
            prediction_file = os.path.join(case_folder, "prediction.nii.gz")
            prediction_image = nib.load(prediction_file)
            prediction = prediction_image.get_data()
            rows.append([self.dice_coefficient(func(truth), func(prediction))for func in masking_functions])

        df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        print(self.config.data_set)
        if os.path.exists("./prediction"):
            df.to_csv("./prediction/"+self.config.data_set+"_scores.csv")
        else:
            os.makedirs("./predicion/")
            df.to_csv("./prediction/" + self.config.data_set + "_scores.csv")

        scores = dict()
        for index, score in enumerate(df.columns):
            values = df.values.T[index]
            scores[score] = values[ pd.isnull(values) == False]

        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel("Dice Coefficient")
        plt.savefig("validation_scores_boxplot.png")
        plt.close()

        if os.path.exists("./training.log"):
            training_df = pd.read_csv("./training.log").set_index('epoch')

            plt.plot(training_df['loss'].values, label='training loss')
            plt.plot(training_df['val_loss'].values, label='validation loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, len(training_df.index)))
            plt.legend(loc='upper right')
            plt.savefig('loss_graph.png')

