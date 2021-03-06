import sys
import numpy as np
import csv
import math, cmath
from collections import deque

dataName = ['phase vector',
            'tag signal',
            'RN16',
            'round']

def log_parser(filename, antNum):
    dataList = []
    with open('data/'+filename) as csvfile:
        logReader = csv.reader(csvfile)
        firstLine = True

        for line in logReader:
            if firstLine:
                #handle first line that has column name
                firstLine = False
            else:
                try:
                    dataVector = []

                    #make phase vector array
                    ant_count = 0
                    phase_vector = []
                    for i in range(antNum):
                        phase_vector.append(cmath.rect(1, math.radians(int(line[i]))))

                    #save in vector
                    dataVector.append(phase_vector)
                    dataVector.append(complex(float(line[-8]), float(line[-7])))
                    dataVector.append(int(line[-2]))
                    dataVector.append(int(line[-1]))

                    dataList.append(dataVector)
                except ValueError:
                    #we came here because of decode failure RN16 record '-'
                    pass
    return dataList

def log_data_filter(dataList, RN16):
    filtered_dataList = []

    for data in dataList:
        if data[3]%2==0 and data[2]==RN16:
            filtered_dataList.append(data)

    return filtered_dataList

def channel_estimator(phaseData, tagSignal):
    phaseData = np.matrix(phaseData)
    tagSignal = np.matrix(tagSignal)

    channel = np.linalg.inv(phaseData) * tagSignal.T
    

    return channel.T.tolist()[0]
                
def data_analyzer(logData, antNum):
    firstPush = antNum
    phaseData = deque([])
    tagSignal = deque([])
    channelList = []

    for data in logData[0:antNum]:
        phaseData.append(data[0])
        tagSignal.append(data[1])

    channelList.append(channel_estimator(phaseData, tagSignal))
    
    for data in logData[antNum:]:
        phaseData.popleft()
        phaseData.append(data[0])
        tagSignal.popleft()
        tagSignal.append(data[1])

        channelList.append(channel_estimator(phaseData, tagSignal))

    return channelList

def channel_analyzer(channelList, antNum):
    antData = []
    for i in range(antNum):
        antData.append([])

    for data in channelList:
        for i in range(antNum):
            antData[i].append(data[i])

    antAvg = []
    antStd = []
    for ant in antData:
        antAvg.append(np.mean(np.array(ant)))
        antStd.append(np.std(np.array(ant)))

    print(antAvg)
    print(antStd)

if __name__ == '__main__':
    dataList = log_data_filter(log_parser('log.csv',4), 0x5555)
    channelList = data_analyzer(dataList, 4)
    channel_analyzer(channelList, 4)
