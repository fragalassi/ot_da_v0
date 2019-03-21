import numpy as np
import keras
from keras import backend as K

class compute_gamma(keras.callbacks.Callback):

    def on_batch_begin(self, batch, logs=None):
        pred = self.model.predict(batch)
        K.set_value(self.gamma, self.)