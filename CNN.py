#!/usr/bin/env python3

import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.core import Reshape

import movie

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
        # self.train_x /= 255
        # self.test_x /= 255

        # DONE: one hot encoding for train_y and test_y

        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        cats = len(test_y[0])

        # DONE: build you CNN model
        act='relu'
        self.model = Sequential()
        self.model.add(Conv2D(32, 3, strides=3, padding='same', activation=act, input_shape=(268,182,3,)))
        self.model.add(Conv2D(64, 3, strides=3, padding='same', activation=act))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, 3, strides=3, padding='same', activation=act))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, 3, strides=3, padding='same', activation=act))
        self.model.add(Flatten())
        # self.model.add(Dropout(0.25))
        self.model.add(Dense(2048, activation=act))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(512, activation=act))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(cats, activation='sigmoid'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

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
    mc.create_data_arrays()
    print('created data arrays')

    cnn = CNN(mc.x_train[:args.limit], mc.y_train[:args.limit], mc.x_test, mc.y_test, epochs=50, batch_size=128)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
