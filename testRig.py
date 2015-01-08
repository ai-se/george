from __future__ import division,print_function
import sys, random, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from lib import *
from where2 import *
from interpolation import *
from extrapolation import *
from coc81 import *
from JPL import *
import sk


DUPLICATION_SIZE = 0
CLUSTERER = launchWhere2
GET_CLUSTER = leaf
#DUPLICATOR = extrapolateNTimes
DUPLICATOR = interpolateNTimes
DO_TUNE = False
MODEL = JPL


"""
Creates a generator of 1 test record 
and rest training records
"""
def loo(dataset):
  for index,item in enumerate(dataset):
    yield item, dataset[:index]+dataset[index+1:]

"""
### Printing Stuff
Print without newline:
Courtesy @timm
"""
def say(*lst): 
  print(*lst,end="")
  sys.stdout.flush()

def formatForCART(test,trains,
        indep=lambda x: x.cells[:-3]
        ,dep=lambda x: x.cells[-3]):
  trainInputSet = []
  trainOutputSet = []
  for train in trains:
    trainInputSet+=[indep(train)]
    trainOutputSet+=[dep(train)]
  return trainInputSet, trainOutputSet, indep(test), dep(test)

"""
Effort is the third column in nasa93 dataset
"""
def effort(row):
    return row.cells[-3]

"""
Selecting the closest cluster and the closest row
""" 
def clusterk1(score, duplicatedModel, tree, test, desired_effort):
  test_leaf = GET_CLUSTER(duplicatedModel, test, tree)
  nearest_row = closest(duplicatedModel, test, test_leaf.val)
  test_effort = effort(nearest_row)
  error = abs(desired_effort - test_effort)/desired_effort
  score += error
  
"""
Selecting K-nearest neighbors and finding the mean
expected effort
"""
def kNearestNeighbor(score, duplicatedModel, test, desired_effort, k=1):
  nearestN = closestN(duplicatedModel, k, test, duplicatedModel._rows)
  expectedSum = sum(map(lambda x:effort(x[1]), nearestN))
  test_effort = expectedSum / k
  score += abs(desired_effort - test_effort)/desired_effort  

"""
Classification and Regression Trees from sk-learn
"""
def CART(score, cartIP, test, desired_effort):
  trainIp, trainOp, testIp, testOp = formatForCART(test,cartIP);
  #print(len(trainIp))
  decTree = DecisionTreeClassifier(criterion="entropy", random_state=1)
  decTree.fit(trainIp,trainOp)
  test_effort = decTree.predict(testIp)[0]
  #print(test_effort, desired_effort)
  score += abs(desired_effort - test_effort)/desired_effort
  
def testRig(dataset=nasa93(doTune=DO_TUNE), duplicator=interpolateNTimes, clstrByDcsn = None):
  rseed(1)
  scores=dict(clstr=N(),CARTT=N())
  #scores=dict(clstr=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(test)
    duplicatedModel = duplicator(dataset, train, CLUSTERER, DUPLICATION_SIZE, clstrByDcsn)
    if (clstrByDcsn == None):
      tree = CLUSTERER(duplicatedModel, rows=None, verbose=False)
    else :
      tree = CLUSTERER(duplicatedModel, rows=None, verbose=False, clstrByDcsn = clstrByDcsn)
    if DUPLICATION_SIZE == 0:
      cartIP = train
    else:
      cartIP = duplicatedModel._rows
    n = scores["clstr"]
    n.go and clusterk1(n, duplicatedModel, tree, test, desired_effort)
    #n = scors[k"]
    #n.go and kNearestNeighbor(n, duplicatedModel, test, desired_effort, k=3)
    n = scores["CARTT"]
    n.go and CART(n, cartIP, test, desired_effort)
  return scores
  


def testDriver():
  skData = [];
  
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( no tuning )         "] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=True, weighKLOC=False),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( Tuning KLOC )       "] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=True),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( Weighing Norm KLOC )"] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False, sdivWeigh = True),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( Weights using sdiv )"] + n.cache.all)
  
  global CLUSTERER
  CLUSTERER = launchWhereV3
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR,clstrByDcsn=False)
  for key,n in scores.items():
    skData.append([key+"( 1st level obj )     "] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR,clstrByDcsn=True)
  for key,n in scores.items():
    skData.append([key+"( 2nd level obj )     "] + n.cache.all)
    
  print("")
  sk.rdivDemo(skData)
  
testDriver()

def testKLOCWeighDriver():
  dataset = MODEL(doTune=False, weighKLOC=True)
  tuneRatio = 0.9
  skData = [];
  while(tuneRatio <= 1.2):
    dataset.tuneRatio = tuneRatio
    scores = testRig(dataset=dataset,duplicator=DUPLICATOR)
    for key,n in scores.items():
      skData.append([key+"( "+str(tuneRatio)+" )"] + n.cache.all)
    tuneRatio += 0.01
  print("")
  sk.rdivDemo(skData)

#testKLOCWeighDriver()

def testKLOCTuneDriver():
  tuneRatio = 0.9
  skData = [];
  while(tuneRatio <= 1.2):
    dataset = MODEL(doTune=True, weighKLOC=False, klocWt=tuneRatio)
    scores = testRig(dataset=dataset,duplicator=DUPLICATOR)
    for key,n in scores.items():
      skData.append([key+"( "+str(tuneRatio)+" )"] + n.cache.all)
    tuneRatio += 0.01
  print("")
  sk.rdivDemo(skData)
  
#testKLOCTuneDriver()
