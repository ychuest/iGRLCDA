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

def partition(ls, size):
    return [ls[i:i+size] for i in range(0, len(ls), size)]



NewAllEdgeNum = []
ReadMyCsv(NewAllEdgeNum, "LMSNPLncMiNum.csv")
print('NewAllEdgeNum[0]', NewAllEdgeNum[0])
print(len(NewAllEdgeNum))



RandomList = random.sample(range(0, len(NewAllEdgeNum)), len(NewAllEdgeNum))
print('len(RandomList)', len(RandomList))
NewRandomList = partition(RandomList, math.ceil(len(RandomList) / 5))
print('len(NewRandomList[0])', len(NewRandomList[0]))
StorFile(NewRandomList, 'NewRandomList.csv')

