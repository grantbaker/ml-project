#!/usr/bin/env python3


import movie
import time

mc = movie.MovieContainer()
#mc.add_csv_file('test.csv')
mc.add_csv_file('data/MovieGenre.csv')
print('added csv')
mc.remove_movies_without_posters()
print('removed without files')
mc.remove_different_size_images()
print('removed different sizes')
mc.create_cat_vecs()
print('created cat vecs')
#mc.load_images_into_mem()
mc.create_data_arrays()
print(mc.x_train[2])
print(mc.y_train[2])

time.sleep(10)

