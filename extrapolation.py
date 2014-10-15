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

def launchExtrapolate(m, dataset):
  extrapolate(m, dataset)
  dataList = list(dataset.dataset)
  dataList = [list(dataList[i])for i in range(len(dataList))]
  return data(indep=INDEP, less= LESS, _rows=dataList)
  
def getLeafNodes(tree):
  leafNodes = []
  for node in leaves(tree):
    leafNodes.append(node)
  return leafNodes

def extrapolate(m, dataset):
  tree = launchWhere2(m, TREE_VERBOSE)
  leaf_nodes = getLeafNodes(tree)
  if len(leaf_nodes)>0:
    max_extrapolation = 2*len(m._rows)
    extrapolationCount = 0
    while extrapolationCount < max_extrapolation:
      rClusters = random.sample(leaf_nodes,2)
      generateDuplicates(rClusters[0][0].val, rClusters[1][0].val, dataset)
      extrapolationCount += 1

def generateDuplicates(clusterA, clusterB, dataset):
  randChoice = random.choice([1,2])
  randElements = random.sample(clusterA,randChoice) + random.sample(clusterB, 3-randChoice)
  dataset.dataset.add(createHybrid(randElements))
    
def createHybrid(originals):
  def mergeData(index):
    return x[index] + rNum *(y[index] - z[index])
  rNum = random.random()
  x = originals[0].cells
  y = originals[1].cells
  z = originals[2].cells 
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

def extrapolateNTimes(initialData, extrapolationCount=1):
  # extrapolates the dataset to 2^(extrapolationCount)
  timesExtrapolated = 0
  while timesExtrapolated < extrapolationCount:
    dataset = ExtendedDataset()
    initialData = launchExtrapolate(initialData, dataset)
    timesExtrapolated += 1
  print len(dataset)
  launchWhere2(initialData, True)
  return initialData

@go
def _extrapolate():
  extrapolateNTimes(nasa93(), 2)
