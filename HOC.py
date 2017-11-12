from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse
import movie
import csv
import os.path

from random import shuffle

DATA_LOCATION = os.path.join('data','download')
CSVF='data/MovieGenre (copy).csv'
def getMovies(csvfile):
    with open(csvfile) as f:
            csv_reader = csv.reader(f, delimiter=',')
            print(csv_reader)
            a=[]
            b=[]
            for row in csv_reader:
                a.append(row[0])
                b.append(row[4])
    return a,b



def HOC(path):
        image = cv2.imread(path)
        chans = cv2.split(image)
        colors = ("b", "g", "r")
        n=0;
        hoc=np.zeros((3,32))
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            hist=hist/sum(hist)
            for i in range(32):
                for j in range(8):
                    hoc[n,i]+=hist[8*i+j]
            hoc[n,:]=hoc[n,:]/sum(hoc[n,:])
            n=n+1
        return hoc

class HOCmovie():
    def __init__(self,id,gen):
        self.id=id
        self.gen=gen
        self.hoc=HOC(DATA_LOCATION+'/'+id+'.jpg')
        self.flhoc=self.hoc.flatten()

mc = movie.MovieContainer()
IDs,genres=getMovies(CSVF)
hocs=[]
gens=[]
for i in range(1,len(IDs)):
    gen=genres[i]
    id=IDs[i]
    try:
        a=HOCmovie(id,gen)
        c=1
    except Exception as e:
        c=-1
        print(-1)
    if c==1:
        hocs.append(a.flhoc)
        gens.append(a.gen)
print(np.shape(hocs))


catlist = []
for key in self.movies.keys():
    for cat in self.movies[key].genres:
        if cat not in catlist:
            catlist.append(cat)
catlist.sort()
for key in self.movies.keys():
    self.movies[key].catvec = np.zeros(len(catlist))
        for cat in self.movies[key].genres:
        self.movies[key].catvec[catlist.index(cat)] = 1
