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
import Tool

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




LMSNPLncMiNum = []
ReadMyCsv(LMSNPLncMiNum, "LMSNPLncMiNum.csv")
print('LMSNPLncMiNum[0]', LMSNPLncMiNum[0])
print('len(LMSNPLncMiNum)', len(LMSNPLncMiNum))

AllLncNum = []
ReadMyCsv(AllLncNum, "AllLncNum.csv")

AllMiNum = []
ReadMyCsv(AllMiNum, "AllMiNum.csv")

print(AllMiNum[0])

import Tool
NegativeSample = Tool.NegativeGenerate(LMSNPLncMiNum, AllLncNum, AllMiNum)
Tool.StorFile(NegativeSample, 'NegativeSample.csv')
