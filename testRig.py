from __future__ import division,print_function
import sys, random, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LinearRegression
from lib import *
from where2 import *
from interpolation import *
from extrapolation import *
from coc81 import *
from JPL import *
from coc2010 import *
import sk
import TEAK
import CoCoMo

DUPLICATION_SIZE = 0
CLUSTERER = launchWhere2
GET_CLUSTER = leaf
#DUPLICATOR = extrapolateNTimes
DUPLICATOR = interpolateNTimes
DO_TUNE = False
MODEL = coc2010

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
Selecting the closest cluster and the closest row
""" 
def clusterk1(score, duplicatedModel, tree, test, desired_effort):
  test_leaf = GET_CLUSTER(duplicatedModel, test, tree)
  nearest_row = closest(duplicatedModel, test, test_leaf.val)
  test_effort = effort(nearest_row)
  error = abs(desired_effort - test_effort)/desired_effort
  #print("clusterk1", test_effort, desired_effort, error)
  score += error

"""
Performing LinearRegression inside a cluster
to estimate effort
"""
def linRegressCluster(score, duplicatedModel, tree, test, desired_effort):
  
  def getTrainData(rows):
    trainIPs, trainOPs = [], []
    for row in rows:
      #trainIPs.append(row.cells[:len(duplicatedModel.indep)])
      trainIPs.append([row.cosine])
      trainOPs.append(effort(row))
    return trainIPs, trainOPs
  
  def fastMapper(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    data = test_leaf.val
    one  = any(data)             
    west = furthest(duplicatedModel,one,data, what = what)  
    east = furthest(duplicatedModel,west,data, what = what)
    c    = dist(duplicatedModel,west,east, what = what)
    test_leaf.west, test_leaf.east, test_leaf.c = west, east, c
    for one in data:
      a = dist(duplicatedModel,one,west, what = what)
      b = dist(duplicatedModel,one,east, what = what)
      x = (a*a + c*c - b*b)/(2*c) # cosine rule
      one.cosine = x
      
  def getCosine(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    a = dist(duplicatedModel,test,test_leaf.west, what = what)
    b = dist(duplicatedModel,test,test_leaf.east, what = what)
    return (a*a + test_leaf.c**2 - b*b)/(2*test_leaf.c) # cosine rule
    
  test_leaf = GET_CLUSTER(duplicatedModel, test, tree)
  fastMapper(test_leaf)
  trainIPs, trainOPs = getTrainData(test_leaf.val)
  clf = LinearRegression()
  clf.fit(trainIPs, trainOPs)
  test_effort = clf.predict(getCosine(test_leaf))
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
  scores=dict(clstr=N(),CARTT=N(), lRgCl=N())
  #scores=dict(clstr=N(), lRgCl=N())
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
    n = scores["lRgCl"]
    n.go and linRegressCluster(n, duplicatedModel, tree, test, desired_effort)
    n = scores["CARTT"]
    n.go and CART(n, cartIP, test, desired_effort)
  return scores
  
"""
Test Rig to test CoCoMo for
a particular dataset
"""
def testCoCoMo(dataset=nasa93(), a=2.94, b=0.91):
  scores = dict(COCOMO2 = N(), COCONUT= N())
  tuned_a, tuned_b = CoCoMo.coconut(dataset, dataset._rows)
  for score in scores.values():
    score.go=True
  for row in dataset._rows:
    say('.')
    desired_effort = effort(row)
    test_effort = CoCoMo.cocomo2(dataset, row.cells, a, b)
    test_effort_tuned = CoCoMo.cocomo2(dataset, row.cells, tuned_a, tuned_b)
    scores["COCOMO2"] += abs(desired_effort - test_effort)/desired_effort
    scores["COCONUT"] += abs(desired_effort - test_effort_tuned)/desired_effort
  return scores

def testDriver():
  skData = [];

  scores = testCoCoMo(dataset=MODEL())
  for key, n in scores.items():
    skData.append([key+".                  ."] + n.cache.all)
  
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( no tuning          )"] + n.cache.all)
  
  scores = testRig(dataset=MODEL(doTune=True, weighKLOC=False),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( Tuning KLOC        )"] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=True),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( Weighing Norm KLOC )"] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False, sdivWeigh = 1),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( sdiv_weight **  1  )"] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False, sdivWeigh = 2),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( sdiv_weight **  2  )"] + n.cache.all)
  
  global CLUSTERER
  CLUSTERER = TEAK.teak
  global GET_CLUSTER
  GET_CLUSTER = TEAK.leafTeak
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False, sdivWeigh = 1),duplicator=DUPLICATOR)
  for key,n in scores.items():
    skData.append([key+"( TEAK               )"] + n.cache.all)
  
  '''
  global CLUSTERER
  CLUSTERER = launchWhereV3
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR,clstrByDcsn=False)
  for key,n in scores.items():
    skData.append([key+"( 1st level obj )     "] + n.cache.all)
    
  scores = testRig(dataset=MODEL(doTune=False, weighKLOC=False),duplicator=DUPLICATOR,clstrByDcsn=True)
  for key,n in scores.items():
    skData.append([key+"( 2nd level obj )     "] + n.cache.all)
  '''
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

#testRig(dataset=MODEL(doTune=False, weighKLOC=False), duplicator=interpolateNTimes, clstrByDcsn = None)