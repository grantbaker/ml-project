from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse
import movie
import csv
import os.path

from random import shuffle

DATA_LOCATION = os.path.join('data','download')

'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
cv2.imshow("image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
i=0
hoc=np.zeros((256,256))
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    print(color)
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
    print(np.size(hist))
    hoc[:,i]=hist.reshape((256,1))
    i=i+1
print ("flattened feature vector size: ",  (np.array(features).flatten().shape))
print(np.size(hoc))
#plt.show()'''

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
IDs,genres=getMovies('data/MovieGenre (copy).csv')
print(genres)
print(np.shape(IDs))
hocs=[]
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
print(np.shape(hocs))
