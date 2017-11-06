#!/usr/bin/env python3

import os
import urllib.request

import movie

class MoviePosterDownloader():

    def __init__(self, csvfile, output_dir):
        self.movie_container = movie.MovieContainer()
        self.movie_container.add_csv_file(csvfile)
        self.movie_container.remove_movies_without_links()

        self.output_dir = output_dir

    def download(self):
        removekeys = []
        i = 0
        l = len(self.movie_container.movies.keys())
        for m in self.movie_container.movies.keys():
            i += 1
            mov = self.movie_container.movies[m]
            print('Downloading: {} of {} | {} | {}'.format(i, l, m, mov.title))
            if os.path.isfile(os.path.join(self.output_dir, str(m) + '.jpg')):
                print('Already downloaded: {} | {}'.format(m, mov.title))
            else:
                try:
                    urllib.request.urlretrieve(
                        mov.poster_link,
                        os.path.join(self.output_dir, str(m) + '.jpg')
                    )
                except Exception as e:
                    print(e)
                    removekeys.append(m)

        for key in removekeys:
            del self.movie_container.movies[key]

if __name__ == '__main__':
    mpd = MoviePosterDownloader('data/MovieGenre.csv', 'data/download')
    mpd.download()
