### MNIST
from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
import h5py
'''
sono partito con 1 cnn e kernel 3, 3 * 8 filtri
con 32 filtri faccio peggio
con stride (1, 1) faccio peggio (!)
non ho sottratto la media

ora:
kernel      = (5, 5)
filters     = 16
image_size  = (28, 28)
image_shape = (1, image_size[0], image_size[1])
pool_size   = (2, 2)
strides     = (2, 2)
dense       = 128

su kaggle: 72° con 0.99657 (nel test set: 0.9854)




'''


kernel      = (5, 5)
filters     = 16
image_size  = (28, 28)
image_shape = (1, image_size[0], image_size[1])
pool_size   = (2, 2)
strides     = (2, 2)
dense       = 128

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = (60000, 28, 28)
# y_train = (60000,)
# X_test = (10000, 28, 28)
# y_test = (10000,)
X_train = X_train.reshape(X_train.shape[0], 1, image_size[0], image_size[1])
y_train = np_utils.to_categorical(y_train, 10)

X_test = X_test.reshape(X_test.shape[0], 1, image_size[0], image_size[1])
y_test = np_utils.to_categorical(y_test, 10)


#model = load_model('mnist-59.h5')

model = Sequential()
model.add(Convolution2D(filters, kernel[0], kernel[1], border_mode='same', input_shape=image_shape))
model.add(Activation('relu'))
model.add(Convolution2D(filters, kernel[0], kernel[1], border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size, strides, border_mode = 'same'))
model.add(Flatten())
model.add(Dense(dense))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for iteration in range(1, 59):
    print('Iteration: ', iteration)
    model.fit(X_train, y_train, batch_size=128, nb_epoch=1)
    if iteration > 55 or iteration % 10 == 0:
      model.save("mnist-{0}.h5".format(iteration))

ret = model.evaluate(X_test, y_test, batch_size = 32, verbose = 1)
print(ret)
