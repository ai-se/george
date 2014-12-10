from __future__ import division,print_function
import sys, random, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from lib import *
from where2 import *
from interpolation import *
from extrapolation import *

DUPLICATION_SIZE = 3
CLUSTERER = launchWhere2
GET_CLUSTER = leaf
#DUPLICATOR = extrapolateNTimes
DUPLICATOR = interpolateNTimes
DO_TUNE = False


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

def testRig(dataset=nasa93(doTune=DO_TUNE), duplicator=interpolateNTimes):
  rseed(1)
  def effort(row):
    return row.cells[-3]
  scores=dict(clusterk1=N(),knn=N(),CART=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(test)
    duplicatedModel = duplicator(dataset, train, DUPLICATION_SIZE) 
    tree = CLUSTERER(duplicatedModel, rows=None, verbose=False)
    """
    Selecting the closest cluster and the closest row
    """ 
    def clusterk1(score):
      test_leaf = GET_CLUSTER(duplicatedModel, test, tree)
      nearest_row = closest(duplicatedModel, test, test_leaf.val)
      test_effort = effort(nearest_row)
      error = abs(desired_effort - test_effort)/desired_effort
      score += error
    """
    Selecting K-nearest neighbors and finding the mean
    expected effort
    """
    def kNearestNeighbor(score, k=1):
      nearestN = closestN(duplicatedModel, k, test, duplicatedModel._rows)
      expectedSum = sum(map(lambda x:effort(x[1]), nearestN))
      test_effort = expectedSum / k
      score += abs(desired_effort - test_effort)/desired_effort
    
    def CART(score):
      if DUPLICATION_SIZE == 0:
        cartIP = train
      else:
        cartIP = duplicatedModel._rows
      trainIp, trainOp, testIp, testOp = formatForCART(test,cartIP);
      #print(len(trainIp))
      decTree = DecisionTreeClassifier(criterion="entropy", random_state=1)
      decTree.fit(trainIp,trainOp)
      test_effort = decTree.predict(testIp)[0]
      #print(test_effort, desired_effort)
      score += abs(desired_effort - test_effort)/desired_effort

    n = scores["clusterk1"]
    n.go and clusterk1(n)
    n = scores["knn"]
    n.go and kNearestNeighbor(n, k=1)
    n = scores["CART"]
    n.go and CART(score)
  #print(scores["clusterk1"])
  print("")
  for key,n in scores.items():
    n.go and print(key,
        ":median",n.cache.has().median,
        ":has",n.cache.has().iqr)
        
testRig(duplicator=DUPLICATOR)
