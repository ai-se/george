from __future__ import division,print_function
import sys, random, math
from lib import *
from where2 import *


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


def testRig(dataset=nasa93()):
  def effort(row):
      return row.cells[-3]
  scores=dict(clusterk1=N(),knn=N())
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(test)
    tree = launchWhere2(dataset, train, verbose=False)
    """
    Selecting the closest cluster and the closest row
    """
    def clusterk1(score):
      test_leaf = leaf(dataset, test, tree)
      nearest_row = closest(dataset, test, test_leaf.val)
      test_effort = effort(nearest_row)
      score += abs(desired_effort - test_effort)/desired_effort
    """
    Selecting K-nearest neighbors and finding the mean
    expected effort
    """
    def kNearestNeighbor(score, k=1):
      nearestN = closestN(dataset, k, test, train)
      expectedSum = sum(map(lambda x:effort(x[1]), nearestN))
      test_effort = expectedSum / k
      score += abs(desired_effort - test_effort)/desired_effort
    n = scores["clusterk1"]
    n.go and clusterk1(n)
    n = scores["knn"]
    n.go and kNearestNeighbor(n)
  print("")
  for key,n in scores.items():
    n.go and print(key,
        ":median",n.cache.has().median,
        ":has",n.cache.has().iqr)
        
testRig()
