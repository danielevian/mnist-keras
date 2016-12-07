from __future__ import print_function
import pandas
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

data = pandas.read_csv('/Users/daniele/Downloads/test.csv')
data = data.values.reshape(data.shape[0], 1, 28, 28)

model = load_model('mnist-0.9854.h5')
classes = model.predict_classes(data)
df = pandas.DataFrame(data = np.array(classes), columns=["Label"])
df.index += 1
df.to_csv('submission.csv', index_label='ImageId')
