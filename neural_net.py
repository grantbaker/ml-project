#!/usr/bin/env python3

import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers.core import Reshape

# for f_score calculation
from keras import backend as K


# for sml loss function
from tensorflow.contrib.sparsemax import sparsemax_loss, sparsemax
from tensorflow.python.ops import math_ops

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# for inception
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

import movie
import numpy as np

def f_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_pos / (possible_pos + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_pos / (predicted_pos + K.epsilon())
        return precision

    beta = 1

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return (1+beta**2)*((precision*recall)/((beta**2)*precision+recall))

def sml(labels,logits):
    sm=sparsemax(logits)
    #loss = -np.dot(logits,labels)
    #smz=sparsemax(logits)


    shifted_logits = logits - \
        math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

    # sum over support
    support = math_ops.cast(sm > 0, sm.dtype)
    sum_s = support * sm * (shifted_logits - 0.5 * sm)

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - shifted_logits)

    return math_ops.reduce_sum(sum_s + q_part, axis=1)

class INCEPTION:

    def __init__(self, train_x, train_y, test_x, test_y, epochs=15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epochs = epochs

        # DONE: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length



        # # normalize data to range [0, 1]
        # train_x = train_x / 255
        # test_x = test_x / 255

        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        cats = len(test_y[0])
        imsize = np.shape(train_x[0])
        print(imsize)

        input_tensor = Input(shape=(imsize[0],imsize[1],imsize[2],))

        inception_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        x = inception_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(8192, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(8192, activation='relu')(x)
        genres = Dense(cats, activation='sigmoid')(x)

        self.model = Model(inputs=input_tensor, outputs=genres)

        for layer in inception_model.layers[:249]:
            layer.trainable = False
        for layer in inception_model.layers[249:]:
            layer.trainable = True

        self.model.compile(loss=sml,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy', f_score])


    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''

        self.model.fit(self.train_x, self.train_y,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=1,
          validation_data=(self.test_x, self.test_y))

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc


class CNN:
    '''
    CNN classifier
    '''

    def __init__(self, train_x, train_y, test_x, test_y, epochs = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epochs = epochs

        # DONE: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length



        # # normalize data to range [0, 1]
        # train_x = train_x / 255
        # test_x = test_x / 255

        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        cats = len(test_y[0])

        # DONE: build you CNN model
        act='relu'
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(268,182,3,)))
        self.model.add(Conv2D(64, 3, strides=3, padding='same', activation=act))
        self.model.add(Conv2D(64, 3, strides=3, padding='same', activation=act))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, 3, strides=3, padding='same', activation=act))
        self.model.add(Conv2D(128, 3, strides=3, padding='same', activation=act))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, 3, strides=3, padding='same', activation=act))
        self.model.add(Conv2D(512, 3, strides=3, padding='same', activation=act))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation=act))
        #self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation=act))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(128, activation=act))
        self.model.add(Dense(cats, activation='sigmoid'))

        self.model.compile(loss=sml,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy', f_score])

    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''

        self.model.fit(self.train_x, self.train_y,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=1,
          validation_data=(self.test_x, self.test_y))

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    mc = movie.MovieContainer()
    mc.add_csv_file('data/MovieGenre.csv')
    print('added csv')
    mc.remove_movies_without_posters()
    print('removed without files')
    mc.remove_different_size_images()
    print('removed different sizes')
    mc.create_cat_vecs()
    print('created cat vecs')
    mc.create_data_arrays(test_proportion=0.2)
    print('created data arrays')

    cnn = INCEPTION(mc.x_train[:args.limit], mc.y_train[:args.limit], mc.x_test, mc.y_test, epochs=1, batch_size=128)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)

    print_size = 5
    evals = cnn.model.predict(mc.x_test[:print_size],batch_size=print_size)
    for i in range(print_size):
        evals_genres = list(zip(evals[i],mc.genre_list))
        evals_genres.sort(reverse=True)
        print(evals[i])
        print(evals_genres[0:2])
        print(mc.x_test_filenames[i])
        #print(mc.y_test[i], ' | ', evals[i])
