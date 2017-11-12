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
            (csv_reader)
            a=[]
            b=[]
            for row in csv_reader:
                a.append(row[0])
                b.append(row[4].split('|'))
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
            #hoc[n,:]=hoc[n,:]/sum(hoc[n,:])
            n=n+1
        return hoc

class HOCmovie():
    def __init__(self,id,gen):
        self.id=id
        self.gen=gen
        self.catvec=[]
        self.hoc=HOC(DATA_LOCATION+'/'+id+'.jpg')
        self.flhoc=self.hoc.flatten()

class HOCmovieList():
    def __init__(self,path):
        IDs,genres=getMovies(path)
        self.IDs=[]
        self.movies = dict()
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
                self.movies[id]=a
                self.IDs.append(id)
            catlist = []
            for key in self.IDs:
                for cat in self.movies[key].gen:
                    if cat not in catlist:
                        catlist.append(cat)
        catlist.sort()
        self.cats=catlist
        for key in self.IDs:
            self.movies[key].catvec = np.zeros(len(catlist))
            for cat in self.movies[key].gen:
                self.movies[key].catvec[catlist.index(cat)] = 1

    def GenData(self,test_proportion=0.3):
        key_list=self.IDs
        split_i = int(len(key_list)*test_proportion)
        test_list = key_list[0:split_i]
        train_list = key_list[split_i:]
        movies=self.movies
        self.x_test = np.zeros((len(test_list), len(movies[test_list[0]].flhoc)), dtype=np.int8)
        self.y_test = np.zeros((len(test_list), self.movies[test_list[0]].catvec.shape[0]), dtype=np.int8)
        self.x_train = np.zeros((len(train_list), len(movies[train_list[0]].flhoc)), dtype=np.int8)
        self.y_train = np.zeros((len(train_list), self.movies[train_list[0]].catvec.shape[0]), dtype=np.int8)
        i=0
        for j in test_list:
            self.x_test[i]=(self.movies[j].flhoc)
            self.y_test[i]=(self.movies[j].catvec)
            i+=1
        i=0
        for j in train_list:
            self.x_train[i]=(self.movies[j].flhoc)
            self.y_train[i]=(self.movies[j].catvec)
            i+=1

lis=HOCmovieList(CSVF)
lis.GenData()
