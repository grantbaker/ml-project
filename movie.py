#!/usr/bin/env python3

import csv
import os.path
import numpy as np

from scipy.misc import imread
from random import shuffle

DATA_LOCATION = os.path.join('data','download')

class Movie():

    def __init__(self, csvline):
        #print(csvline)

        # csvline is of the following format:
        # csvline = [imdbId, imdb_link, title, score, genres, poster_link]

        self.id = int(csvline[0])
        self.imdb_link = csvline[1]
        title = csvline[2]

        # separate year from title
        if title[-1] == ')':
            self.title = title[:-7]
            self.year = int(title[-5:-1])
        else:
            self.title = title
            self.year = 0

        self.score = csvline[3]
        self.genres = csvline[4].split('|')
        self.poster_link = csvline[5]
        self.actors = csvline[9].split('|')
        self.director = csvline[10]

        self.catvec = -1
        self.matrix = -1


class MovieContainer():

    def __init__(self):
        self.movies = dict()

        self.images_loaded = False

        self.x_train_images = -1
        self.x_train_actor_names = []
        self.x_train_directors = []
        self.x_test_images = -1
        self.x_test_actor_names = []
        self.x_test_directors = []
        self.y_test_labels = []
        self.x_test_filenames = []
        self.y_train = -1
        self.y_test = -1
        self.genre_list = []

    def add_csv_file(self, csvfile):
        with open(csvfile) as f:
            csv_reader = csv.reader(f, delimiter=',')

            num_rows = -1
            for row in csv_reader:
                if num_rows == -1:
                    num_rows += 1
                    #print(num_rows)
                    continue
                else:
                    num_rows += 1
                    #print(num_rows)
                    try:
                        key = int(row[0])
                        if key not in self.movies.keys():
                            self.movies[key] = Movie(row)
                    except Exception as e:
                        print(row)

    def remove_movies_without_links(self):
        removekeys = []
        for key in self.movies.keys():
            if len(self.movies[key].poster_link) < 4:
                removekeys.append(key)

        for key in removekeys:
            del self.movies[key]

    def remove_movies_without_posters(self):
        removekeys = []
        for key in self.movies.keys():
            filename = str(key) + '.jpg'
            if not os.path.isfile(os.path.join(DATA_LOCATION,filename)):
                removekeys.append(key)

        for key in removekeys:
            del self.movies[key]

    def create_cat_vecs(self):
        catlist = []
        for key in self.movies.keys():
            for cat in self.movies[key].genres:
                if cat not in catlist:
                    catlist.append(cat)

        catlist.sort()
        self.genre_list = catlist[:]

        for key in self.movies.keys():
            self.movies[key].catvec = np.zeros(len(catlist))

            for cat in self.movies[key].genres:
                 self.movies[key].catvec[catlist.index(cat)] = 1

    def load_images_into_mem(self):
        for key in self.movies.keys():
            filename = str(key) + '.jpg'
            self.movies[key].matrix = imread(os.path.join(DATA_LOCATION, filename), mode='RGB')

        self.images_loaded = True

    def remove_images_from_mem(self):
        for key in self.movies.keys():
            self.movies[key].matrix = -1

        self.images_loaded = False

    def remove_bw_images(self):
        # TODO: Finish this
        removekeys = []
        for key in self.movies.keys():
            filename = str(key) + '.jpg'
            im = imread(os.path.join(DATA_LOCATION, filename))

    def remove_different_size_images(self):
        shapes = dict()
        for key in self.movies.keys():
            filename = str(key) + '.jpg'
            im = imread(os.path.join(DATA_LOCATION, filename), mode='RGB')
            sh = im.shape
            if sh not in shapes.keys():
                shapes[sh] = []
            shapes[sh].append(key)

        max_shape = max(shapes, key= lambda x: len(set(shapes[x])))

        num_deleted = 0
        for sh in [shape for shape in shapes.keys() if shape != max_shape]:
            for key in shapes[sh]:
                del self.movies[key]
                num_deleted += 1
        print('Deleted: ', num_deleted)

    def create_data_arrays(self,test_proportion=0.5):
        key_list = list(self.movies.keys())
        shuffle(key_list)
        split_i = int(len(key_list)*test_proportion)
        test_list = key_list[0:split_i]
        train_list = key_list[split_i:]

        #print(test_list, train_list)

        im_size = imread(os.path.join(DATA_LOCATION, str(test_list[0]) + '.jpg')).shape

        self.x_test_images = np.zeros((len(test_list), im_size[0], im_size[1], im_size[2]), dtype=np.int8)
        self.y_test = np.zeros((len(test_list), self.movies[test_list[0]].catvec.shape[0]), dtype=np.int8)
        i = 0
        for key in test_list:
            if self.images_loaded:
                self.x_test_images[i] = self.movies[key].matrix
            else:
                filename = str(key) + '.jpg'
                self.x_test_images[i] = imread(os.path.join(DATA_LOCATION, filename), mode='RGB')
                self.x_test_filenames.append(filename)
            self.x_test_actor_names.append(self.movies[key].actors)
            self.x_test_directors.append(self.movies[key].director)
            self.y_test[i] = self.movies[key].catvec
            self.y_test_labels.append(self.movies[key].genres)
            i += 1

        self.x_train_images = np.zeros((len(train_list), im_size[0], im_size[1], im_size[2]), dtype=np.int8)
        self.y_train = np.zeros((len(train_list), self.movies[train_list[0]].catvec.shape[0]), dtype=np.int8)
        i = 0
        for key in train_list:
            if self.images_loaded:
                self.x_train_images[i] = self.movies[key].matrix
            else:
                filename = str(key) + '.jpg'
                self.x_train_images[i] = imread(os.path.join(DATA_LOCATION, filename), mode='RGB')
            self.x_train_actor_names.append(self.movies[key].actors)
            self.x_train_directors.append(self.movies[key].director)
            self.y_train[i] = self.movies[key].catvec
            i += 1


# mc = MovieContainer()
# mc.add_csv_file('data/MetaData2.csv')
# print('added csv')
# mc.remove_movies_without_posters()
# print('removed without files')
# mc.remove_different_size_images()
# print('removed different sizes')
# mc.create_cat_vecs()
# print('created cat vecs')
# mc.create_data_arrays(test_proportion=0.2)
# print('created data arrays')
