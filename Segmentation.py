from numpy import *
import numpy as np
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


PositiveSample = []
ReadMyCsv(PositiveSample, "PositiveSample.csv")
print('PositiveSample[0]', PositiveSample[0])
print(len(PositiveSample))

NewRandomList = []
ReadMyCsv(NewRandomList, "NewRandomList.csv")
print('NewRandomList[0]', NewRandomList[0])
print(len(NewRandomList))

AllEdgeNum = []
ReadMyCsv(AllEdgeNum, "AllEdgeNum.csv")
print('AllEdgeNum[0]', AllEdgeNum[0])
print(len(AllEdgeNum))


counter = 0
while counter < len(NewRandomList):

    Num = 0
    TestListPair = []
    TrainListPair = []
    counter2 = 0
    while counter2 < len(NewRandomList):
        if counter2 == counter:
            TestListPair.extend(PositiveSample[Num:Num + len(NewRandomList[counter2])])
        if counter2 != counter:
            TrainListPair.extend(PositiveSample[Num:Num + len(NewRandomList[counter2])])
        Num = Num + len(NewRandomList[counter2])
        counter2 = counter2 + 1

    TestName = 'TestName' + str(counter) + '.csv'
    StorFile(TestListPair, TestName)
    TrainName = 'TrainName' + str(counter) + '.csv'
    StorFile(TrainListPair, TrainName)

    NewTrainName = 'NewTrainName' + str(counter) + '.csv'
    TrainListPair.extend(AllEdgeNum)
    StorFile(TrainListPair, NewTrainName)

    counter = counter + 1
