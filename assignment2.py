from math import floor
import math
from time import time
from turtle import color
from cv2 import threshold
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal;


def emgDecomposition(signal, tWindow):
    rectifiedSignal = rectifySignal(signal)
    noise = np.array(calculateNoise(rectifiedSignal))
    signalThreshold = np.std(noise) * 3
    detectedMuaps = {}
    detectedMuapsIndices = {}
    detectedMus = {}
    i=0
    while i +tWindow< len(rectifiedSignal):
        muap = []
        muapIndices=[]
        if(np.average(rectifiedSignal[i:i+tWindow])>signalThreshold):
            y = [rectifiedSignal[j] for j in range(i,i+tWindow)]
            muap.extend(y)
            x = [j for j in range(i,i+tWindow)]
            muapIndices.extend(x)
            i+=tWindow
            while(i< len(rectifiedSignal) and np.average(rectifiedSignal[i:i+tWindow+1])>signalThreshold):
                muap.append(rectifiedSignal[i])
                muapIndices.append(i)
                i+=1
        if(len(muap)>0):
            detectedMuaps[str(detectedMuaps.__len__()+1)]= muap
            detectedMuapsIndices[str(detectedMuapsIndices.__len__()+1)]= muapIndices
            adjustedMuap =  adjustMuap(muapIndices,muap,tWindow,signal)
            muTemplateIndex = findTemplate(detectedMus,adjustedMuap)
            if(detectedMus.keys().__contains__(muTemplateIndex)):
                detectedMus[muTemplateIndex] = [ ((detectedMus[muTemplateIndex][j] + adjustedMuap[j] )/ 2) for j in range(len(adjustedMuap)) ]
            else:
                detectedMus[muTemplateIndex] = adjustedMuap
        i+=1
    returnDict = {"rectifiedSignal":rectifiedSignal,"detectedMuaps":detectedMuaps,"detectedMuapsIndices":detectedMuapsIndices,"detectedMus":detectedMus}
    return returnDict

def findTemplate(detectedMus,adjustedMuap):
    dffth = math.pow(12.65,5)
    for key in detectedMus.keys():
        templateMu = detectedMus[key]
        difference =0
        for i in range(len(templateMu)):
          difference+= math.pow((templateMu[i] - adjustedMuap[i]),2)
        if(difference < dffth):
            return key
    return str(detectedMus.__len__()+1)

def adjustMuap(indices,muap,tWindow,signal):
    maxindex = indices[0]
    maxval = muap[0]
    for i in range(len(muap)) :
        if(muap[i]>=maxval):
            maxval=muap[i]
            maxindex=indices[i]
    returnMUap= [signal[j] for j in range(maxindex-(tWindow//2),maxindex+(tWindow//2))]
    return returnMUap

def calculateNoise(signal):
    initialThreshold =  np.std(signal[0:100]) * 3
    noise = []
    for i in range(signal.shape[0]):
        if(signal[i] <= initialThreshold):
            noise.append(signal[i])
    return noise

def rectifySignal(signal):
    for i in range(len(signal)):
        signal[i] = abs(signal[i])
    return signal


muData = np.loadtxt("Data.txt")
fig1 = plt.figure(figsize=(20,10))
# plt = plt.subplots()
returnData = emgDecomposition(muData,20)
rectifiedSignal = returnData["rectifiedSignal"]
detectedMuapsIndices=returnData["detectedMuapsIndices"]
detectedMuaps =returnData["detectedMuaps"]
t = np.arange(len(rectifiedSignal))
plt.plot(t[30000:35000],rectifiedSignal[30000:35000]) 
t2 =[]
muap=[]
for key in  detectedMuapsIndices.keys():
    indices = detectedMuapsIndices[key]
    muap1 = detectedMuaps[key]
    maxindex = indices[0]
    maxval = muap1[0]
    for i in range(len(muap1)) :
        if(muap1[i]>maxval):
            maxval=muap1[i]
            maxindex=indices[i]
    if(t[30000:35000].__contains__(maxindex)):
        t2.append(maxindex)
        muap.append(maxval)
    # print(t2)
    # print(muap)
plt.scatter(t2,muap, marker="*",color="red")
plt.savefig("DetectedMUAP.jpg")


fig = plt.figure(figsize=(20,20))
ourColors={"1":"red","2":"black","3":"green"}
for key in returnData['detectedMus'].keys():
        ax1 = fig.add_subplot(int("31{}".format(key)))
        muap = returnData['detectedMus'][key]
        t = np.arange(len(muap))
        # print(muap)
        # print(t)
        ourColor = "blue"
        if(ourColors.__contains__(key)):
            ourColor = ourColors[key]
        ax1.plot(t,muap,color=ourColor)
plt.savefig("Templates.jpg")
        


