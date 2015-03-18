"""
# https://code.google.com/p/promisedata/source/browse/#svn%2Ftrunk%2Feffort%2Falbrecht

Standard header:

"""
from __future__ import division,print_function
import  sys
sys.dont_write_bytecode = True
from lib import *

"""
@attribute	Syear	numeric
@attribute	App	numeric
@attribute	Har	numeric
@attribute	Dba	numeric
@attribute	Ifc	numeric
@attribute	Source	numeric
@attribute	Telonuse 	numeric
@attribute	Nlan	numeric
@attribute	T01	numeric
@attribute	T02	numeric
@attribute	T03 	numeric
@attribute	T04	numeric
@attribute	T05	numeric
@attribute	T06 	numeric
@attribute	T07	numeric
@attribute	T08 	numeric
@attribute	T09 	numeric
@attribute	T10	numeric
@attribute	T11 	numeric
@attribute	T12 	numeric
@attribute	T13 	numeric
@attribute	T14 	numeric
@attribute	T15 	numeric
@attribute	Duration	numeric
@attribute	Size	numeric
@attribute	Time	numeric
@attribute	Effort	numeric
"""

def maxwell(weighFeature = False, 
           split = "median"):
  vl=1;l=2;n=3;h=4;vh=5;xh=6;_=0
  return data(indep= [ 
     # 0..6
    'Syear','App','Har','Dba','Ifc','Source','Telonuse',
    'Nlan','T01','T02','T03','T04','T05','T06',
    'T07','T08','T09','T10','T11','T12','T13',
    'T14','T15','Duration','Size','Time'],
    less = ['Effort'],
    _rows=[
      [92,2,2,1,2,2,0,3,4,3,5,3,3,3,4,5,4,5,4,4,4,4,5,16,647,8,7871],
      [93,2,2,1,2,2,0,3,2,3,3,3,3,3,2,2,4,3,4,4,4,4,4,5,130,9,845],
      [90,1,2,1,2,2,0,2,3,3,2,3,3,4,2,3,4,5,4,3,2,3,3,8,254,6,2330],
      [86,3,2,1,2,2,0,3,2,2,4,2,2,1,3,5,4,4,5,4,3,2,3,16,1056,2,21272],
      [88,2,2,1,2,2,0,2,3,3,3,4,3,3,4,3,4,4,3,4,5,4,4,12,383,4,4224],
      [92,2,3,1,2,2,1,4,2,3,3,3,3,3,2,2,4,4,4,4,4,5,4,12,345,8,2826],
      [87,2,2,1,2,2,0,2,4,3,5,4,3,2,3,5,5,5,3,4,4,2,3,27,209,3,7320],
      [86,2,2,1,2,2,0,1,2,3,3,2,2,2,4,5,4,3,3,3,3,2,3,24,366,2,9125],
      [87,2,4,2,2,1,0,2,4,3,3,2,1,2,4,5,3,2,2,2,3,4,2,54,1181,3,11900],
      [87,1,2,1,2,2,0,2,2,3,2,3,3,3,2,5,3,4,2,3,2,3,3,13,181,3,4300],
      [90,2,5,1,2,1,0,1,5,3,4,2,3,1,3,3,3,2,2,2,1,1,2,21,739,6,4150],
      [91,3,1,0,2,2,0,2,2,2,2,4,3,3,1,4,4,3,4,4,1,5,1,7,108,7,900],
      [90,3,5,0,2,2,0,3,2,3,3,4,2,2,2,4,4,3,5,3,3,4,2,10,48,6,583],
      [91,1,2,1,2,2,0,2,2,3,2,4,3,3,3,5,4,3,3,4,2,4,3,19,249,7,2565],
      [92,2,2,1,2,2,0,2,3,4,3,3,3,3,3,3,5,5,2,4,3,3,3,11,371,8,4047],
      [87,2,2,1,2,2,0,1,2,3,2,4,3,3,4,4,4,3,2,4,3,3,3,13,211,3,1520],
      [91,2,2,1,2,2,1,4,4,1,3,3,3,4,4,5,4,4,4,4,3,3,4,32,1849,7,25910],
      [89,2,2,1,2,2,1,4,4,3,4,3,4,4,5,4,5,4,5,5,3,1,4,38,2482,5,37286],
      [85,3,3,1,2,2,0,4,3,2,3,3,3,2,4,5,4,4,4,4,4,2,3,40,434,1,15052],
      [87,2,2,1,2,2,0,3,4,3,4,4,4,2,3,4,5,5,3,4,4,2,3,29,292,3,11039],
      [90,3,3,1,2,2,0,4,4,4,2,3,3,3,4,3,4,4,5,3,2,3,3,14,2954,6,18500],
      [91,2,3,1,2,2,0,1,4,3,2,4,3,3,4,4,5,4,3,4,2,4,3,14,304,7,9369],
      [89,2,5,1,2,1,0,1,4,3,2,3,2,3,2,4,4,4,2,2,3,3,3,28,353,5,7184],
      [92,2,2,1,2,2,1,4,2,2,2,4,3,3,4,4,5,5,5,4,2,2,4,16,567,8,10447],
      [91,2,2,1,2,2,1,3,4,3,4,3,3,3,3,3,4,4,3,3,2,3,3,13,467,7,5100],
      [87,2,2,1,2,2,0,3,4,3,3,4,3,3,4,5,4,4,4,4,4,2,4,45,3368,3,63694],
      [92,3,2,1,2,2,1,2,3,3,3,4,3,4,2,4,4,3,2,4,4,2,4,4,253,8,1651],
      [91,4,3,1,2,2,1,3,1,4,2,3,3,4,2,3,2,3,2,4,2,4,3,10,196,7,1450],
      [92,1,2,1,2,2,0,4,3,4,2,3,3,4,3,5,3,3,3,4,4,5,3,12,185,8,1745],
      [88,3,2,1,2,2,0,2,2,4,3,2,2,2,3,4,5,4,4,4,3,3,4,6,387,4,1798],
      [88,5,2,1,2,2,0,2,1,3,3,3,3,2,4,5,3,2,3,3,3,4,3,28,430,4,2957],
      [89,2,2,1,2,2,0,2,3,4,2,3,3,3,3,4,3,4,2,4,3,3,3,6,204,5,963],
      [88,2,2,1,2,2,0,1,3,3,3,4,3,3,2,3,4,2,3,4,2,4,3,6,71,4,1233],
      [91,1,3,1,1,1,0,3,4,2,4,3,5,3,4,5,5,4,3,5,4,4,5,6,840,7,3240],
      [90,2,3,1,1,1,0,4,4,2,4,3,5,3,5,3,5,5,4,5,3,4,5,11,1648,6,10000],
      [91,2,3,1,1,1,0,4,4,2,4,3,5,3,5,3,5,5,4,5,3,4,5,8,1035,7,6800],
      [85,3,2,1,2,2,0,1,3,3,4,2,3,2,3,4,3,3,3,4,4,3,3,22,548,1,3850],
      [91,3,3,1,2,2,1,3,4,3,4,4,4,3,3,4,4,4,5,5,4,3,4,31,2054,7,14000],
      [88,5,2,1,2,2,0,2,2,3,2,4,3,3,3,4,3,3,3,3,2,4,3,26,302,4,5787],
      [93,3,3,1,2,2,1,3,4,2,2,3,4,4,4,2,4,2,4,3,4,3,4,22,1172,9,9700],
      [91,1,5,1,1,2,0,1,3,3,3,2,2,3,3,3,4,3,3,4,4,4,3,7,253,7,1100],
      [92,2,2,1,2,2,0,2,3,4,2,4,3,3,3,3,5,5,3,3,2,3,3,14,227,8,5578],
      [92,2,2,1,2,2,0,3,4,3,3,3,3,3,2,3,4,3,3,4,3,3,3,6,59,8,1060],
      [91,1,2,1,2,2,1,4,3,3,4,4,4,4,3,4,4,3,5,4,3,2,3,6,299,7,5279],
      [89,3,2,1,2,2,0,1,3,3,3,4,3,3,3,3,5,3,4,4,3,2,4,15,422,5,8117],
      [90,2,5,1,2,1,0,3,4,4,3,5,4,4,5,3,4,4,3,5,2,4,4,9,1058,6,8710],
      [90,1,5,4,2,2,0,3,4,2,2,3,3,2,4,4,4,4,3,4,4,5,3,9,65,6,796],
      [88,3,3,1,2,2,0,3,5,5,3,3,2,3,4,5,5,4,4,4,3,4,4,26,390,4,11023],
      [90,2,2,1,2,2,0,2,4,4,2,3,4,3,2,2,3,3,3,3,2,4,4,13,193,6,1755],
      [91,1,2,1,2,2,1,4,4,3,2,3,3,3,3,3,4,3,4,4,4,3,3,28,1526,7,5931],
      [93,3,3,1,2,2,1,2,2,3,3,3,3,3,4,2,4,3,4,4,2,3,3,13,575,9,4456],
      [87,2,2,1,2,2,0,1,2,3,3,4,3,3,3,4,4,3,2,4,4,2,3,13,509,3,3600],
      [88,5,2,1,2,2,0,3,1,4,4,2,3,3,2,3,2,2,2,4,5,3,3,12,583,4,4557],
      [88,3,2,1,2,2,0,2,4,3,5,3,3,3,4,5,4,3,3,4,3,3,3,14,315,4,8752],
      [89,3,2,1,2,2,0,2,3,4,5,3,3,3,3,3,4,4,2,4,4,3,3,12,138,5,3440],
      [88,2,3,1,2,2,0,3,3,4,3,3,3,3,3,4,4,4,4,4,4,4,3,9,257,4,1981],
      [85,2,2,1,2,2,0,1,2,3,3,2,2,2,4,5,4,3,3,4,4,2,3,30,423,1,13700],
      [91,5,5,1,2,1,0,3,4,2,4,3,3,3,3,3,5,3,3,4,3,4,4,20,495,7,7105],
      [90,3,3,1,2,2,1,4,2,3,3,4,3,3,3,4,4,3,4,3,2,4,3,16,622,6,6816],
      [92,1,2,1,2,2,1,2,3,3,3,3,3,4,4,5,5,5,5,4,3,2,3,12,204,8,4620],
      [90,3,3,1,2,2,1,4,2,3,2,3,3,2,3,5,5,4,5,5,1,5,4,15,616,6,7451],
      [91,3,3,1,2,2,0,3,2,4,3,3,3,3,4,3,5,5,5,4,4,5,4,33,3643,7,39479]
    ],
    _tunings =[[
    #         vlow  low   nom   high  vhigh xhigh
    #scale factors:
    'Prec',   6.20, 4.96, 3.72, 2.48, 1.24, _ ],[
    'Flex',   5.07, 4.05, 3.04, 2.03, 1.01, _ ],[
    'Resl',   7.07, 5.65, 4.24, 2.83, 1.41, _ ],[
    'Pmat',   7.80, 6.24, 4.68, 3.12, 1.56, _ ],[
    'Team',   5.48, 4.38, 3.29, 2.19, 1.01, _ ]],
    weighFeature = weighFeature,
    _split = split,
    _isCocomo = False,
    ignores=[24]
    )

def _maxwell(): print(maxwell())