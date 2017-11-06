#!/usr/bin/env python3

import csv

class Movie():

    def __init__(self, csvline):
        # print(csvline)

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


class MovieContainer():

    def __init__(self):
        self.movies = dict()

    def add_csv_file(self, csvfile):
        with open(csvfile) as f:
            csv_reader = csv.reader(f, delimiter=',')

            num_rows = -1
            for row in csv_reader:
                if num_rows == -1:
                    num_rows += 1
                    print(num_rows)
                    continue
                else:
                    num_rows += 1
                    print(num_rows)
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
