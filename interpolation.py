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

class InterpolatedDataSet:
  def __init__(i, dataset=None) :
    if (dataset == None) :
      i.dataset = set()
    else :
      i.dataset = None
  def __len__(i):
    return len(i.dataset)

def launchInterpolate(m, dataset):
  tree = launchWhere2(m)
  interpolate(tree, dataset)
  print len(dataset)
  dataList = list(dataset.dataset)
  dataList = [list(dataList[i])for i in range(len(dataList))]
  return data(indep=INDEP, less= LESS, _rows=dataList)

def interpolate(tree, dataset):
  if len(tree._kids) > 0:
    for node in tree._kids:
      interpolate(node, dataset)
  else:
    generateDuplicates(tree.val, dataset)

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
    return x[index] + rNum * max(x[index],y[index]) - min(x[index],y[index])
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

@go
def _interpolate():
  dataset = InterpolatedDataSet()
  print(launchInterpolate(nasa93, dataset))
