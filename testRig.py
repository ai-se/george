from __future__ import division,print_function
import sys, random, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from lib import *
from where2 import *
import Technix.sk as sk
import Technix.CoCoMo as CoCoMo
import Technix.sdivUtil as sdivUtil
from Technix.smote import smote
from Technix.batman import smotify
from Technix.TEAK import teak, leafTeak, teakImproved
from Technix.atlm import lin_reg
from Technix.atlm_pruned import lin_reg_pruned
from Models import *
MODEL = nasa93.nasa93
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

def formatForCART(dataset,test,trains):
  def indep(x):
    rets=[]
    indeps = x.cells[:len(dataset.indep)]
    for i,val in enumerate(indeps):
      if i not in dataset.ignores:
        rets.append(val)
    return rets
  dep   = lambda x: x.cells[len(dataset.indep)]
  trainInputSet = []
  trainOutputSet = []
  for train in trains:
    trainInputSet+=[indep(train)]
    trainOutputSet+=[dep(train)]
  return trainInputSet, trainOutputSet, indep(test), dep(test)

"""
Selecting the closest cluster and the closest row
""" 
def clusterk1(score, duplicatedModel, tree, test, desired_effort, leafFunc):
  test_leaf = leafFunc(duplicatedModel, test, tree)
  nearest_row = closest(duplicatedModel, test, test_leaf.val)
  test_effort = effort(duplicatedModel, nearest_row)
  error = abs(desired_effort - test_effort)/desired_effort
  #print("clusterk1", test_effort, desired_effort, error)
  score += error

def cluster_nearest(model, tree, test, leaf_func):
  test_leaf = leaf_func(model, test, tree)
  nearest_row = closest(model, test, test_leaf.val)
  return effort(model, nearest_row)
  
def clustermean2(score, duplicatedModel, tree, test, desired_effort, leafFunc):
  test_leaf = leafFunc(duplicatedModel, test, tree)
  nearestN = closestN(duplicatedModel, 2, test, test_leaf.val)
  if (len(nearestN)==1) :
    nearest_row = nearestN[0][1]
    test_effort = effort(duplicatedModel, nearest_row)
    error = abs(desired_effort - test_effort)/desired_effort
  else :
    test_effort = sum(map(lambda x:effort(duplicatedModel, x[1]), nearestN[:2]))/2
    error = abs(desired_effort - test_effort)/desired_effort  
  score += error

def cluster_weighted_mean2(model, tree, test, leaf_func):
  test_leaf = leaf_func(model, test, tree)
  nearest_rows = closestN(model, 2, test, test_leaf.val)
  wt_0 = nearest_rows[1][0]/(nearest_rows[0][0] + nearest_rows[1][0] + 0.000001)
  wt_1 = nearest_rows[0][0]/(nearest_rows[0][0] + nearest_rows[1][0] + 0.000001)
  return effort(model, nearest_rows[0][1]) * wt_0 + effort(model, nearest_rows[1][1]) * wt_1

def clusterWeightedMean2(score, duplicatedModel, tree, test, desired_effort, leafFunc):
  test_leaf = leafFunc(duplicatedModel, test, tree)
  nearestN = closestN(duplicatedModel, 2, test, test_leaf.val)
  if (len(nearestN)==1) :
    nearest_row = nearestN[0][1]
    test_effort = effort(duplicatedModel, nearest_row)
    error = abs(desired_effort - test_effort)/desired_effort
  else :
    nearest2 = nearestN[:2]
    wt_0, wt_1 = nearest2[1][0]/(nearest2[0][0]+nearest2[1][0]+0.00001) , nearest2[0][0]/(nearest2[0][0]+nearest2[1][0]+0.00001)
    test_effort = effort(duplicatedModel, nearest2[0][1])*wt_0 + effort(duplicatedModel, nearest2[1][1])*wt_1
    #test_effort = sum(map(lambda x:effort(duplicatedModel, x[1]), nearestN[:2]))/2
    error = abs(desired_effort - test_effort)/desired_effort  
  score += error
  
def clusterVasil(score, duplicatedModel, tree, test, desired_effort, leafFunc, k):
  test_leaf = leafFunc(duplicatedModel, test, tree)
  if k > len(test_leaf.val):
    k = len(test_leaf.val)
  nearestN = closestN(duplicatedModel, k, test, test_leaf.val)
  if (len(nearestN)==1) :
    nearest_row = nearestN[0][1]
    test_effort = effort(duplicatedModel, nearest_row)
    error = abs(desired_effort - test_effort)/desired_effort
  else :
    nearestk = nearestN[:k]
    test_effort, sum_wt = 0,0
    for dist, row in nearestk:
      test_effort += (1/(dist+0.000001))*effort(duplicatedModel,row)
      sum_wt += (1/(dist+0.000001))
    test_effort = test_effort / sum_wt
    error = abs(desired_effort - test_effort)/desired_effort  
  score += error
  
  
  
"""
Performing LinearRegression inside a cluster
to estimate effort
"""
def linRegressCluster(score, duplicatedModel, tree, test, desired_effort, leafFunc=leaf, doSmote=False):
  
  def getTrainData(rows):
    trainIPs, trainOPs = [], []
    for row in rows:
      #trainIPs.append(row.cells[:len(duplicatedModel.indep)])
      trainIPs.append([row.cosine])
      trainOPs.append(effort(duplicatedModel, row))
    return trainIPs, trainOPs
  
  def fastMapper(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    data = test_leaf.val
    #data = smotify(duplicatedModel, test_leaf.val,k=5, factor=100)
    one  = any(data)             
    west = furthest(duplicatedModel,one,data, what = what)  
    east = furthest(duplicatedModel,west,data, what = what)
    c    = dist(duplicatedModel,west,east, what = what)
    test_leaf.west, test_leaf.east, test_leaf.c = west, east, c
    
    for one in data:
      if c == 0:
        one.cosine = 0
        continue
      a = dist(duplicatedModel,one,west, what = what)
      b = dist(duplicatedModel,one,east, what = what)
      x = (a*a + c*c - b*b)/(2*c) # cosine rule
      one.cosine = x
      
  def getCosine(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    if (test_leaf.c == 0):
      return 0
    a = dist(duplicatedModel,test,test_leaf.west, what = what)
    b = dist(duplicatedModel,test,test_leaf.east, what = what)
    return (a*a + test_leaf.c**2 - b*b)/(2*test_leaf.c) # cosine rule
    
  test_leaf = leafFunc(duplicatedModel, test, tree)
  #if (len(test_leaf.val) < 4) :
   # test_leaf = test_leaf._up
  if (len(test_leaf.val)>1) and doSmote:
    data = smote(duplicatedModel, test_leaf.val,k=5, N=100)
    linearRegression(score, duplicatedModel, data, test, desired_effort)
  else :
    fastMapper(test_leaf)
    trainIPs, trainOPs = getTrainData(test_leaf.val)
    clf = LinearRegression()
    clf.fit(trainIPs, trainOPs)
    test_effort = clf.predict(getCosine(test_leaf))
    error = abs(desired_effort - test_effort)/desired_effort
    score += error
  
  
"""
Performing LinearRegression over entire dataset
"""
def linearRegression(score, model, train, test, desired_effort):
  def getTrainData(rows):
    trainIPs, trainOPs = [], []
    for row in rows:
      trainRow=[]
      for i,val in enumerate(row.cells[:len(model.indep)]):
        if i not in model.ignores:
          trainRow.append(val)
      trainIPs.append(trainRow)
      trainOPs.append(effort(model, row))
    return trainIPs, trainOPs
  
  trainIPs, trainOPs = getTrainData(train)
  clf = LinearRegression()
  clf.fit(trainIPs, trainOPs)
  testIP=[]
  for i,val in enumerate(test.cells[:len(model.indep)]):
    if i not in model.ignores:
      testIP.append(val)
  test_effort = clf.predict(testIP)
  error = abs(desired_effort - test_effort)/desired_effort
  score += error

"""
Selecting K-nearest neighbors and finding the mean
expected effort
"""
def kNearestNeighbor(score, duplicatedModel, test, desired_effort, k=1, rows = None):
  if rows == None:
    rows = duplicatedModel._rows
  nearestN = closestN(duplicatedModel, k, test, rows)
  test_effort = sorted(map(lambda x:effort(duplicatedModel, x[1]), nearestN))[k//2]
  score += abs(desired_effort - test_effort)/desired_effort

def knn_1(model, row, rest):
  closest_1 = closestN(model, 1, row, rest)[0][1]
  return effort(model, closest_1)

def knn_3(model, row, rest):
  closest_3 = closestN(model, 3, row, rest)
  a = effort(model, closest_3[0][1])
  b = effort(model, closest_3[1][1])
  c = effort(model, closest_3[2][1])
  return (50*a + 33*b + 17*c)/100

"""
Classification and Regression Trees from sk-learn
"""
def CART(dataset, score, cartIP, test, desired_effort):
  trainIp, trainOp, testIp, testOp = formatForCART(dataset, test,cartIP);
  decTree = DecisionTreeRegressor(criterion="mse", random_state=1)
  decTree.fit(trainIp,trainOp)
  test_effort = decTree.predict(testIp)[0]
  score += abs(desired_effort - test_effort)/desired_effort

def cart(model, row, rest):
  train_ip, train_op, test_ip, test_op = formatForCART(model, row, rest)
  dec_tree = DecisionTreeRegressor(criterion="mse", random_state=1)
  dec_tree.fit(train_ip,train_op)
  return dec_tree.predict([test_ip])[0]
  
def showWeights(model):
  outputStr=""
  i=0
  for wt, att in sorted(zip(model.weights, model.indep)):
    outputStr += att + " : " + str(round(wt,2))
    i+=1
    if i%5==0:
      outputStr += "\n"
    else:
      outputStr += "\t"
  return outputStr.strip()

def loc_dist(m,i,j,
         what = lambda m: m.decisions):
  "Euclidean distance 0 <= d <= 1 between decisions"
  dec_index = what(m)[-1]
  n1 = norm(m, dec_index, i.cells[dec_index])
  n2 = norm(m, dec_index, j.cells[dec_index])
  return abs(n1-n2)

def loc_closest_n(model, n, row, other_rows):
  tmp = []
  for other_row in other_rows:
    if id(row) == id(other_row): continue
    d = loc_dist(model, row, other_row)
    tmp += [(d, other_row)]
  return sorted(tmp)[:n]

def loc_1(model, row, rows):
  closest_1 = loc_closest_n(model, 1, row, rows)[0][1]
  return effort(model, closest_1)

def loc_3(model, row, rows):
  closest_3 = loc_closest_n(model, 3, row, rows)
  a = effort(model, closest_3[0][1])
  b = effort(model, closest_3[1][1])
  c = effort(model, closest_3[2][1])
  return (50*a + 33*b + 17*c)/100

def productivity(model, row, rows):
  loc_index = model.decisions[-1]
  productivities = [effort(model, row)/one.cells[loc_index] for one in rows]
  avg_productivity = sum(productivities)/len(productivities)
  return avg_productivity*row.cells[loc_index]

def testRig(dataset=MODEL(), 
            doCART = False,doKNN = False, doLinRg = False):
  scores=dict(clstr=N(), lRgCl=N())
  if doCART:
    scores['CARTT']=N();
  if  doKNN:
    scores['knn_1'],scores['knn_3'],scores['knn_5'] = N(), N(), N()
  if doLinRg:
    scores['linRg'] = N()
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    tree = launchWhere2(dataset, rows=train, verbose=False)
    n = scores["clstr"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort, leaf)
    n = scores["lRgCl"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort)
    if doCART:
      CART(dataset, scores["CARTT"], train, test, desired_effort)
    if doKNN:
      n = scores["knn_1"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=1, rows=train)
      n = scores["knn_3"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=3, rows=train)
      n = scores["knn_5"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=5, rows=train)
    if doLinRg:
      n = scores["linRg"]
      n.go and linearRegression(n, dataset, train, test, desired_effort)
  return scores

def average_effort(model, rows):
  efforts = []
  for row in rows:
    efforts.append(effort(model, row))
  return sum(efforts)/len(efforts)

def effort_error(actual, computed, average):
  return abs((actual**2 - computed**2)/(actual**2 - average**2))
  #return actual - computed
  #return abs((actual**2 - computed**2)/(actual**2))
  #return abs(actual - computed)/actual
  #return ((actual - computed)/(actual - average))**2

"""
Test Rig to test CoCoMo for
a particular dataset
"""
def testCoCoMo(dataset=MODEL(), a=2.94, b=0.91):
  coc_scores = dict(COCOMO2 = N(), COCONUT= N())
  tuned_a, tuned_b = CoCoMo.coconut(dataset, dataset._rows)
  for score in coc_scores.values():
    score.go=True
  for row, rest in loo(dataset._rows):
    #say('.')
    desired_effort = effort(dataset, row)
    avg_effort = average_effort(dataset, rest)
    test_effort = CoCoMo.cocomo2(dataset, row.cells, a, b)
    test_effort_tuned = CoCoMo.cocomo2(dataset, row.cells, tuned_a, tuned_b)
    #coc_scores["COCOMO2"] += ((desired_effort - test_effort) / (desired_effort - avg_effort))**2
    coc_scores["COCOMO2"] += effort_error(desired_effort, test_effort, avg_effort)
    #coc_scores["COCONUT"] += ((desired_effort - test_effort_tuned) / (desired_effort - avg_effort))**2
    coc_scores["COCONUT"] += effort_error(desired_effort, test_effort_tuned, avg_effort)
  return coc_scores

def pruned_coconut(model, row, rows, row_count, column_ratio, noise=None):
  pruned_rows, columns = CoCoMo.prune_cocomo(model, rows, row_count, column_ratio)
  a_tuned, b_tuned = CoCoMo.coconut(model, pruned_rows, decisions=columns, noise=noise)
  return CoCoMo.cocomo2(model, row.cells, a=a_tuned, b=b_tuned, decisions=columns, noise=noise), pruned_rows


    
def testDriver():
  seed(0)
  skData = []
  split = "median"
  dataset=MODEL(split=split)
  if  dataset._isCocomo:
    scores = testCoCoMo(dataset)
    for key, n in scores.items():
      skData.append([key+".       ."] + n.cache.all)
  scores = testRig(dataset=MODEL(split=split),doCART = True, doKNN=True, doLinRg=True)
  for key,n in scores.items():
    if (key == "clstr" or key == "lRgCl"):
      skData.append([key+"(no tuning)"] + n.cache.all)
    else:
      skData.append([key+".         ."] + n.cache.all)

  scores = testRig(dataset=MODEL(split=split, weighFeature = True), doKNN=True)
  for key,n in scores.items():
      skData.append([key+"(sdiv_wt^1)"] + n.cache.all)
  scores = dict(TEAK=N())
  for score in scores.values():
    score.go=True
  dataset=MODEL(split=split)
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    tree = teak(dataset, rows = train)
    n = scores["TEAK"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort, leafTeak)
  for key,n in scores.items():
      skData.append([key+".          ."] + n.cache.all)
  print("")
  print(str(len(dataset._rows)) + " data points,  " + str(len(dataset.indep)) + " attributes")
  print("")
  sk.rdivDemo(skData)
  #launchWhere2(MODEL())
  
#testDriver()

def testKLOCWeighDriver():
  dataset = MODEL(doTune=False, weighKLOC=True)
  tuneRatio = 0.9
  skData = [];
  while(tuneRatio <= 1.2):
    dataset.tuneRatio = tuneRatio
    scores = testRig(dataset=dataset)
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
    scores = testRig(dataset=dataset)
    for key,n in scores.items():
      skData.append([key+"( "+str(tuneRatio)+" )"] + n.cache.all)
    tuneRatio += 0.01
  print("")
  sk.rdivDemo(skData)
  
#testKLOCTuneDriver()

#testRig(dataset=MODEL(doTune=False, weighKLOC=False), duplicator=interpolateNTimes)

def testOverfit(dataset= MODEL(split="median")):
  skData = [];
  scores= dict(splitSize_2=N(),splitSize_4=N(),splitSize_8=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    tree = launchWhere2(dataset, rows=train, verbose=False, minSize=2)
    n = scores["splitSize_2"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort)
    tree = launchWhere2(dataset, rows=train, verbose=False, minSize=4)
    n = scores["splitSize_4"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort)
    tree = launchWhere2(dataset, rows=train, verbose=False, minSize=8)
    n = scores["splitSize_8"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort)
  
  for key,n in scores.items():
      skData.append([key] + n.cache.all)
  print("")
  sk.rdivDemo(skData)
  
#testOverfit()

def testSmote():
  dataset=MODEL(split="variance", weighFeature=True)
  launchWhere2(dataset, verbose=False)
  skData = [];
  scores= dict(sm_knn_1_w=N(), sm_knn_3_w=N(), CART=N())
  for score in scores.values():
    score.go=True
  
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    clones = smotify(dataset, train,k=5, factor=100)
    n = scores["CART"]
    n.go and CART(dataset, scores["CART"], train, test, desired_effort)
    n = scores["sm_knn_1_w"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 1, clones)
    n = scores["sm_knn_3_w"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 3, clones)
  
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
  if dataset._isCocomo:
    for key,n in testCoCoMo(dataset).items():
      skData.append([key] + n.cache.all)
  
  scores= dict(knn_1=N(), knn_3=N())
  dataset=MODEL(split="variance", weighFeature=True)
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    n = scores["knn_1_w"]
    kNearestNeighbor(n, dataset, test, desired_effort, 1, train)
    n = scores["knn_3_w"]
    kNearestNeighbor(n, dataset, test, desired_effort, 3, train)
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
    
  scores= dict(knn_1_w=N(), knn_3_w=N())
  dataset=MODEL(split="variance")
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    n = scores["knn_1"]
    kNearestNeighbor(n, dataset, test, desired_effort, 1, train)
    n = scores["knn_3"]
    kNearestNeighbor(n, dataset, test, desired_effort, 3, train)
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
    
  print("")
  sk.rdivDemo(skData)
  
def testForPaper(model=MODEL):
  split="median"
  print(model.__name__.upper())
  dataset=model(split=split, weighFeature=False)
  print(str(len(dataset._rows)) + " data points,  " + str(len(dataset.indep)) + " attributes")
  dataset_weighted = model(split=split, weighFeature=True)
  launchWhere2(dataset, verbose=False)
  skData = []
  if dataset._isCocomo:
    for key,n in testCoCoMo(dataset).items():
      skData.append([key] + n.cache.all)
  scores = dict(CART = N(), knn_1 = N(),
                knn_3 = N(), TEAK = N(),
                vasil_2=N(), vasil_3=N(),
                vasil_4=N(), vasil_5=N(),)
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    desired_effort = effort(dataset, test)
    tree = launchWhere2(dataset, rows=train, verbose=False)
    tree_teak = teak(dataset, rows = train)
    #n = scores["LSR"]
    #n.go and linearRegression(n, dataset, train, test, desired_effort)
    n = scores["TEAK"]
    n.go and clusterk1(n, dataset, tree_teak, test, desired_effort, leafTeak)
    n = scores["CART"]
    n.go and CART(dataset, scores["CART"], train, test, desired_effort)
    n = scores["knn_1"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 1, train)
    n = scores["knn_3"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 3, train)
  
  for test, train in loo(dataset_weighted._rows):
    desired_effort = effort(dataset, test)
    tree_weighted, leafFunc = launchWhere2(dataset_weighted, rows=train, verbose=False), leaf
    n = scores["vasil_2"]
    n.go and clusterVasil(n, dataset_weighted, tree_weighted, test, desired_effort,leafFunc,2)
    n = scores["vasil_3"]
    n.go and clusterVasil(n, dataset_weighted, tree_weighted, test, desired_effort,leafFunc,3)
    n = scores["vasil_4"]
    n.go and clusterVasil(n, dataset_weighted, tree_weighted, test, desired_effort,leafFunc,4)
    n = scores["vasil_5"]
    n.go and clusterVasil(n, dataset_weighted, tree_weighted, test, desired_effort,leafFunc,5)
  
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
  
  print("")
  sk.rdivDemo(skData)
  print("");print("")
    
  
def testEverything(model = MODEL):
  split="median"
  print('###'+model.__name__.upper())
  dataset=model(split=split, weighFeature=False)
  print('####'+str(len(dataset._rows)) + " data points,  " + str(len(dataset.indep)) + " attributes")
  dataset_weighted = model(split=split, weighFeature=True)
  launchWhere2(dataset, verbose=False)
  skData = [];
  scores= dict(TEAK=N(), linear_reg=N(), CART=N(),
               wt_linRgCl=N(), wt_clstr_whr=N(),
               linRgCl=N(), clstr_whr=N(),
               t_wt_linRgCl=N(), t_wt_clstr_whr=N(),
               knn_1=N(), wt_knn_1=N(), 
               clstrMn2=N(), wt_clstrMn2=N(), t_wt_clstrMn2=N(),
               PEEKING2=N(), wt_PEEKING2=N(), t_wt_PEEKING2=N(),
               t_clstr_whr=N(), t_linRgCl=N(), t_clstrMn2=N(),t_PEEKING2=N())
  #scores= dict(TEAK=N(), linear_reg=N(), linRgCl=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    #say(".")
    desired_effort = effort(dataset, test)
    tree = launchWhere2(dataset, rows=train, verbose=False)
    tree_teak = teakImproved(dataset, rows = train)
    n = scores["TEAK"]
    n.go and clusterk1(n, dataset, tree_teak, test, desired_effort, leaf)
    n = scores["linear_reg"]
    n.go and linearRegression(n, dataset, train, test, desired_effort)
    n = scores["clstr_whr"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort, leaf)
    n = scores["linRgCl"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort, leaf)
    n = scores["knn_1"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 1, train)
    n = scores["clstrMn2"]
    n.go and clustermean2(n, dataset, tree, test, desired_effort, leaf)
    n = scores["PEEKING2"]
    n.go and clusterWeightedMean2(n, dataset, tree, test, desired_effort, leaf)
    n = scores["CART"]
    n.go and CART(dataset, scores["CART"], train, test, desired_effort)
    
    tree, leafFunc = teakImproved(dataset, rows=train, verbose=False),leaf
    n = scores["t_clstr_whr"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort, leafFunc)
    n = scores["t_linRgCl"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort, leafFunc=leafFunc)
    n = scores["t_clstrMn2"]
    n.go and clustermean2(n, dataset, tree, test, desired_effort, leafFunc)
    n = scores["t_PEEKING2"]
    n.go and clusterWeightedMean2(n, dataset, tree, test, desired_effort, leafFunc)
    
  for test, train in loo(dataset_weighted._rows):
    #say(".")
    desired_effort = effort(dataset_weighted, test)
    
    tree_weighted, leafFunc = launchWhere2(dataset_weighted, rows=train, verbose=False), leaf
    n = scores["wt_clstr_whr"]
    n.go and clusterk1(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["wt_linRgCl"]
    n.go and linRegressCluster(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc=leafFunc)
    n = scores["wt_clstrMn2"]
    n.go and clustermean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["wt_PEEKING2"]
    n.go and clusterWeightedMean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    
    tree_weighted, leafFunc = teakImproved(dataset_weighted, rows=train, verbose=False),leaf
    n = scores["t_wt_clstr_whr"]
    n.go and clusterk1(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["t_wt_linRgCl"]
    n.go and linRegressCluster(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc=leafFunc)
    n = scores["wt_knn_1"]
    n.go and kNearestNeighbor(n, dataset_weighted, test, desired_effort, 1, train)
    n = scores["t_wt_clstrMn2"]
    n.go and clustermean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["t_wt_PEEKING2"]
    n.go and clusterWeightedMean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
  if dataset._isCocomo:
    for key,n in testCoCoMo(dataset).items():
      skData.append([key] + n.cache.all)
  print("\n####Attributes")
  print("```")
  print(showWeights(dataset_weighted))
  print("```\n")
  print("```")
  sk.rdivDemo(skData)
  print("```");print("")

  
def testTeakified(model = MODEL):
  split="median"
  print('###'+model.__name__.upper())
  dataset=model(split=split, weighFeature=False)
  print('####'+str(len(dataset._rows)) + " data points,  " + str(len(dataset.indep)) + " attributes")
  dataset_weighted = model(split=split, weighFeature=True)
  launchWhere2(dataset, verbose=False)
  skData = [];
  scores= dict(linear_reg=N(), CART=N(),
               linRgCl_wt=N(), clstr_whr_wt=N(),
               linRgCl=N(), clstr_whr=N(),
               knn_1=N(), knn_1_wt=N(), 
               clstrMn2=N(), clstrMn2_wt=N(),
               PEEKING2=N(), PEEKING2_wt=N())
  #scores= dict(TEAK=N(), linear_reg=N(), linRgCl=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    #say(".")
    desired_effort = effort(dataset, test)
    tree = teakImproved(dataset, rows=train, verbose=False)
    n = scores["linear_reg"]
    n.go and linearRegression(n, dataset, train, test, desired_effort)
    n = scores["clstr_whr"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort, leaf)
    n = scores["linRgCl"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort, leaf)
    n = scores["knn_1"]
    n.go and kNearestNeighbor(n, dataset, test, desired_effort, 1, train)
    n = scores["clstrMn2"]
    n.go and clustermean2(n, dataset, tree, test, desired_effort, leaf)
    n = scores["PEEKING2"]
    n.go and clusterWeightedMean2(n, dataset, tree, test, desired_effort, leaf)
    n = scores["CART"]
    n.go and CART(dataset, scores["CART"], train, test, desired_effort)
    
  for test, train in loo(dataset_weighted._rows):
    #say(".")
    desired_effort = effort(dataset_weighted, test)
    
    tree_weighted, leafFunc = teakImproved(dataset_weighted, rows=train, verbose=False), leaf
    n = scores["clstr_whr_wt"]
    n.go and clusterk1(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["linRgCl_wt"]
    n.go and linRegressCluster(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc=leafFunc)
    n = scores["clstrMn2_wt"]
    n.go and clustermean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["PEEKING2_wt"]
    n.go and clusterWeightedMean2(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc)
    n = scores["knn_1_wt"]
    n.go and kNearestNeighbor(n, dataset_weighted, test, desired_effort, 1, train)    
    
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
  if dataset._isCocomo:
    for key,n in testCoCoMo(dataset).items():
      skData.append([key] + n.cache.all)  
  
  print("```")
  sk.rdivDemo(skData)
  print("```");print("")
  
  
def runAllModels(test_name):
  models = [nasa93.nasa93, coc81.coc81, Mystery1.Mystery1, Mystery2.Mystery2,
            albrecht.albrecht, kemerer.kemerer, kitchenham.kitchenham,
           maxwell.maxwell, miyazaki.miyazaki, telecom.telecom, usp05.usp05,
           china.china, cosmic.cosmic, isbsg10.isbsg10]
  for model in models:
    test_name(model)
    
def printAttributes(model):
  dataset_weighted = model(split="median", weighFeature=True)
  print('###'+model.__name__.upper())
  print("\n####Attributes")
  print("```")
  print(showWeights(dataset_weighted))
  print("```\n")

def testNoth(model=MODEL):
  dataset_weighted = model(split="median", weighFeature=True)
  launchWhere2(dataset_weighted, verbose=False)
  skData = [];
  scores= dict(t_wt_linRgCl_sm=N(), CART=N())
  #scores= dict(TEAK=N(), linear_reg=N(), linRgCl=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset_weighted._rows):
    desired_effort = effort(dataset_weighted, test)
    tree_weighted, leafFunc = launchWhere2(dataset_weighted, rows=train, verbose=False), leaf
    n = scores["t_wt_linRgCl_sm"]
    n.go and linRegressCluster(n, dataset_weighted, tree_weighted, test, desired_effort, leafFunc, doSmote=True)
    n = scores["CART"]
    n.go and CART(dataset_weighted, scores["CART"], train, test, desired_effort)
  for key,n in scores.items():
    skData.append([key] + n.cache.all)
  print("```")
  sk.rdivDemo(skData)
  print("```");print("")

def test_sec4_1():
  """
  Section 4.1
  Cocomo vs LOC
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = dict(COCOMO2 = N(), COCONUT = N(), LOC1 = N(), LOC3 = N())
    tuned_a, tuned_b = CoCoMo.coconut(model, model._rows)
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      loc1_effort = loc_1(model, row, rest)
      loc3_effort = loc_3(model, row, rest)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["LOC1"] += effort_error(desired_effort, loc1_effort, avg_effort)
      model_scores["LOC3"] += effort_error(desired_effort, loc3_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")

def test_sec4_2_standard():
  """
  Section 4.2
  Cocomo vs TEAK vs PEEKING2
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = dict(COCOMO2 = N(), COCONUT = N(), KNN1 = N(), KNN3 = N(), CART = N())
    tuned_a, tuned_b = CoCoMo.coconut(model, model._rows)
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      knn1_effort = loc_1(model, row, rest)
      knn3_effort = loc_3(model, row, rest)
      cart_effort = cart(model, row, rest)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["KNN1"] += effort_error(desired_effort, knn1_effort, avg_effort)
      model_scores["KNN3"] += effort_error(desired_effort, knn3_effort, avg_effort)
      model_scores["CART"]+= effort_error(desired_effort, cart_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")

def test_sec4_2_newer():
  """
  Section 4.3
  Choice of Statistical Ranking Methods
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = dict(COCOMO2 = N(), COCONUT = N(), TEAK = N(), PEEKING2 = N())
    tuned_a, tuned_b = CoCoMo.coconut(model, model._rows)
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      tree_teak = teakImproved(model, rows=rest, verbose=False)
      teak_effort = cluster_nearest(model, tree_teak, row, leafTeak)
      tree = launchWhere2(model, rows=rest, verbose=False)
      peeking_effort = cluster_weighted_mean2(model, tree, row, leaf)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["TEAK"] += effort_error(desired_effort, teak_effort, avg_effort)
      model_scores["PEEKING2"] += effort_error(desired_effort, peeking_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")

def test_sec4_3():
  """
  Section 4.3
  Choice of Statistical Ranking Methods
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = {
      "COCOMO2" : N(),
      "COCONUT" : N(),
      "COCONUT:c0.25,r4" : N(),
      "COCONUT:c0.25,r8" : N(),
      "COCONUT:c0.5,r4" : N(),
      "COCONUT:c0.5,r8" : N(),
      "COCONUT:c1,r4" : N(),
      "COCONUT:c1,r8" : N(),
    }
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      tuned_a, tuned_b = CoCoMo.coconut(model, rest)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 4, 0.25)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.25,r4"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.25)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.25,r8"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 4, 0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r4"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 4, 1)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c1,r4"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 1)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c1,r8"] += effort_error(desired_effort, pruned_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")

def test_sec4_4():
  """
  Section 4.4
  COCOMO with Incorrect Size Estimates
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = {
      "COCOMO2" : N(),
      "COCOMO2:n/2" : N(),
      "COCOMO2:n/4" : N(),
      "COCONUT:c0.5,r8" : N(),
      "COCONUT:c0.5,r8,n/2" : N(),
      "COCONUT:c0.5,r8,n/4" : N(),
    }
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=0.5)
      model_scores["COCOMO2:n/2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=0.25)
      model_scores["COCOMO2:n/4"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,n/2"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=0.25)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,n/4"] += effort_error(desired_effort, pruned_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")


def test_baseline():
  """
  Section 4.3
  Choice of Statistical Ranking Methods
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = dict(COCOMO2 = N(),
                        COCONUT = N(),
                        TEAK = N(),
                        PEEKING2 = N(),
                        BASELINE = N())
    tuned_a, tuned_b = CoCoMo.coconut(model, model._rows)
    for score in model_scores.values():
      score.go=True
    actuals = []
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      actuals.append(desired_effort)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      tree_teak = teakImproved(model, rows=rest, verbose=False)
      teak_effort = cluster_nearest(model, tree_teak, row, leafTeak)
      tree = launchWhere2(model, rows=rest, verbose=False)
      peeking_effort = cluster_weighted_mean2(model, tree, row, leaf)
      baseline_effort = lin_reg(model, row, rest)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["TEAK"] += effort_error(desired_effort, teak_effort, avg_effort)
      model_scores["PEEKING2"] += effort_error(desired_effort, peeking_effort, avg_effort)
      model_scores["BASELINE"] += effort_error(desired_effort, baseline_effort, avg_effort)
    print("### %s"%model_fn.__name__)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")
    # var_actuals = np.var(actuals)
    # for key, n in model_scores.items():
    #   var_model = np.var(n.cache.all)
    #   sk_data.append((var_model/var_actuals,key))
    # sk_data = sorted(sk_data)
    # print("```")
    # line = "----------------------------------------------------"
    # print ('%4s , %22s ,    %s' % \
    #              ('rank', 'name', 'error')+ "\n"+ line)
    # for index, (error, key) in enumerate(sk_data):
    #   print("%4d , %22s ,   %0.4f"%(index+1, key, error))
    # print("```");print("")

def test_pruned_baseline():
  """
  Section 4.4
  COCOMO with Incorrect Size Estimates
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = {
      "COCOMO2" : N(),
      "COCONUT" : N(),
      "BASELINE" : N(),
      "P_BASELINE" : N(),
      "CART" : N()
    }
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      a_tuned, b_tuned = CoCoMo.coconut(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, a_tuned, b_tuned)
      baseline_effort = lin_reg(model, row, rest)
      baseline_pruned_effort = lin_reg_pruned(model, row, rest)
      cart_effort = cart(model, row, rest)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["BASELINE"] += effort_error(desired_effort, baseline_effort, avg_effort)
      model_scores["P_BASELINE"] += effort_error(desired_effort, baseline_pruned_effort, avg_effort)
      model_scores["CART"] += effort_error(desired_effort, cart_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")


def test_pruned_baseline_continuous():
  """
  Section 4.4
  COCOMO with Incorrect Size Estimates
  :param model:
  :return:
  """
  models = [albrecht.albrecht, kitchenham.kitchenham, maxwell.maxwell, miyazaki.miyazaki, china.china]
  for model_fn in models:
    model = model_fn()
    model_scores = {
      "BASELINE" : N(),
      "P_BASELINE" : N(),
      "CART" : N(),
      "TEAK": N()
    }
    for score in model_scores.values():
      score.go=True
    print("### %s"%model_fn.__name__)
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      baseline_effort = lin_reg(model, row, rest)
      baseline_pruned_effort = lin_reg_pruned(model, row, rest)
      cart_effort = cart(model, row, rest)
      tree_teak = teakImproved(model, rows=rest, verbose=False)
      teak_effort = cluster_nearest(model, tree_teak, row, leafTeak)
      model_scores["BASELINE"] += effort_error(desired_effort, baseline_effort, avg_effort)
      model_scores["P_BASELINE"] += effort_error(desired_effort, baseline_pruned_effort, avg_effort)
      model_scores["CART"] += effort_error(desired_effort, cart_effort, avg_effort)
      model_scores["TEAK"] += effort_error(desired_effort, teak_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")


def test_sec_kloc_error():
  """
  Section 4.4
  COCOMO with Incorrect Size Estimates
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = {
      "COCOMO2" : N(),
      "COCOMO2:n/2" : N(),
      "COCOMO2:n/4" : N(),
      "COCOMO2:2*n" : N(),
      "COCOMO2:4*n" : N(),
      "COCONUT:c0.5,r8" : N(),
      "COCONUT:c0.5,r8,n/2" : N(),
      "COCONUT:c0.5,r8,n/4" : N(),
      "COCONUT:c0.5,r8,2*n" : N(),
      "COCONUT:c0.5,r8,4*n" : N(),
    }
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=0.5)
      model_scores["COCOMO2:n/2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=0.25)
      model_scores["COCOMO2:n/4"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=2)
      model_scores["COCOMO2:2*n"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells, noise=4)
      model_scores["COCOMO2:4*n"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=0.5)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,n/2"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=0.25)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,n/4"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=2)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,2*n"] += effort_error(desired_effort, pruned_effort, avg_effort)
      pruned_effort, pruned_rows = pruned_coconut(model, row, rest, 8, 0.5, noise=4)
      avg_effort = average_effort(model, pruned_rows)
      model_scores["COCONUT:c0.5,r8,4*n"] += effort_error(desired_effort, pruned_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")


def test_sec4_1_productivity():
  """
  Updated Section 4.1
  Cocomo vs LOC vs Productivity
  :param model:
  :return:
  """
  models = [Mystery1.Mystery1, Mystery2.Mystery2, nasa93.nasa93, coc81.coc81]
  for model_fn in models:
    model = model_fn()
    model_scores = dict(COCOMO2 = N(), COCONUT = N(),
                        LOC1 = N(), LOC3 = N(),
                        PROD = N())
    tuned_a, tuned_b = CoCoMo.coconut(model, model._rows)
    for score in model_scores.values():
      score.go=True
    for row, rest in loo(model._rows):
      #say('.')
      desired_effort = effort(model, row)
      avg_effort = average_effort(model, rest)
      cocomo_effort = CoCoMo.cocomo2(model, row.cells)
      coconut_effort = CoCoMo.cocomo2(model, row.cells, tuned_a, tuned_b)
      loc1_effort = loc_1(model, row, rest)
      loc3_effort = loc_3(model, row, rest)
      prod_effort = productivity(model, row, rest)
      model_scores["COCOMO2"] += effort_error(desired_effort, cocomo_effort, avg_effort)
      model_scores["COCONUT"] += effort_error(desired_effort, coconut_effort, avg_effort)
      model_scores["LOC1"] += effort_error(desired_effort, loc1_effort, avg_effort)
      model_scores["LOC3"] += effort_error(desired_effort, loc3_effort, avg_effort)
      model_scores["PROD"] += effort_error(desired_effort, prod_effort, avg_effort)
    sk_data = []
    for key, n in model_scores.items():
      sk_data.append([key] + n.cache.all)
    print("### %s"%model_fn.__name__)
    print("```")
    sk.rdivDemo(sk_data)
    print("```");print("")


if __name__ == "__main__":
  #testEverything(albrecht.albrecht)
  #runAllModels(testEverything)
  #testNoth(MODEL)
  seed()
  #test_sec4_4()
  #test_baseline()
  #test_pruned_baseline_continuous()
  #test_sec_kloc_error()
  #test_sec4_1_productivity()
  test_sec4_1()


