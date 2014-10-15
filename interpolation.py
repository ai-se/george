"""
Source file to perform interpolation on the nasa93 data
"""

import sys
import random
from lib import *
from nasa93 import *
from settings import *
from where2 import *
from utils import *
import numpy as np

TREE_VERBOSE=False;

def launchInterpolate(m, dataset):
  tree = launchWhere2(m,TREE_VERBOSE)
  interpolate(tree, dataset)
  dataList = list(dataset.dataset)
  dataList = [list(dataList[i])for i in range(len(dataList))]
  return data(indep=INDEP, less= LESS, _rows=dataList)

def interpolate(tree, dataset):
  leaf_nodes = leaves(tree)
  for node in leaf_nodes:
    generateDuplicates(node[0].val, dataset)

def generateDuplicates(rows, dataset):
  maxSampling = 2 * len(rows)
  sampleIndex = 0
  #max_cols, min_cols = getMinMax(rows)
  while (sampleIndex < maxSampling) :
    sampleIndex += 1
    randomSamples = random.sample(rows,2)
    dataset.dataset.add(createHybrid(randomSamples))
    
def createHybrid(originals):
  def mergeData(index):
    return x[index] + rNum * (max(x[index],y[index]) - min(x[index],y[index]))
  rNum = random.random()
  x = originals[0].cells
  y = originals[1].cells
  hybrid = [mergeData(i) for i in range(len(x))]
  return tuple(formatDuplicate(hybrid))

def formatDuplicate(record):
  formatedRecord = []
  for i in range(len(record)):
    if DATATYPES[i] == int:
      formatedRecord.append(int(record[i]))
    else :
      formatedRecord.append(round(record[i],4))
  return formatedRecord

def getMinMax(rows):
  matrix = np.matrix(rows)
  max_cols = matrix.max(axis=0)
  min_cols = matrix.min(axis=0)
  return max_cols, min_cols

def interpolateNTimes(initialData, interpolationCount=1):
  # interpolates the dataset to 2^(interpolationCount)
  timesInterpolated = 0
  while timesInterpolated < interpolationCount:
    dataset = ExtendedDataset()
    initialData = launchInterpolate(initialData, dataset)
    timesInterpolated += 1
  print len(dataset)
  launchWhere2(initialData,True)
  return initialData

@go
def _interpolate():
  interpolateNTimes(nasa93(), 2)
