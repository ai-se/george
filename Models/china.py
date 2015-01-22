"""
# http://code.google.com/p/promisedata/wiki/china

Standard header:

"""
from __future__ import division,print_function
import  sys
sys.dont_write_bytecode = True
from lib import *

"""
@attribute AFP numeric
@attribute Input numeric
@attribute Output numeric
@attribute Enquiry numeric
@attribute File numeric
@attribute Interface numeric
@attribute Added numeric
@attribute Changed numeric
@attribute Deleted numeric
@attribute PDR_AFP numeric
@attribute PDR_UFP numeric
@attribute NPDR_AFP numeric
@attribute NPDU_UFP numeric
@attribute Resource numeric
@attribute Dev.Type numeric
@attribute Duration numeric
@attribute Effort numeric
@attribute N_effort numeric
"""

def china(doTune = False, weighKLOC=False, 
           klocWt=None, sdivWeigh = None, 
           split = "median"):
  vl=1;l=2;n=3;h=4;vh=5;xh=6;_=0
  return data(indep= [ 
     # 0..15
     'AFP','Input','Output','Enquiry','File','Interface','Added','Changed','Deleted',
     'PDR_AFP', 'PDR_UFP', 'NPDR_AFP', 'NPDU_UFP', 'Resource', 'Dev.Type', 'Duration'],
    less = ['Effort', 'N_effort'],
    _rows=[
      [1587,774,260,340,128,0,1502,0,0,4.7,5,4.7,5,4,0,4,7490,7490],
      [260,9,4,3,193,41,51,138,61,16,16.6,16,16.6,2,0,17,4150,4150],
      [152,25,33,28,42,35,163,0,0,4.4,4.1,4.4,4.1,1,0,9,668,668],
      [252,151,28,8,39,0,69,153,4,12.8,14.3,15.5,17.3,1,0,4,3238,3901],
      [292,93,0,194,20,0,0,307,0,10.3,9.8,12.4,11.7,1,0,13,2994,3607],
      [83,63,0,24,0,0,0,87,0,16.1,15.3,19.3,18.5,1,0,4,1333,1606],
      [79,24,0,23,30,0,0,77,0,20.3,20.9,24.5,25.1,1,0,6,1607,1936],
      [97,0,108,7,0,5,120,0,0,11.9,9.7,11.9,9.7,2,0,7,1158,1158],
      [116,0,23,58,14,20,81,34,0,10.7,10.8,12.9,13,1,0,6,1243,1498],
      [52,39,7,0,0,0,0,46,0,64.8,73.3,78.1,88.3,1,0,7,3372,4063],
      [465,209,129,24,83,15,460,0,0,21.9,22.2,21.9,22.2,1,0,9,10200,10200],
      [67,32,5,16,7,0,25,35,0,25.4,28.4,30.6,34.2,1,0,7,1704,2053],
      [199,0,115,57,0,42,214,0,0,13.3,12.3,15.2,14.2,2,0,7,2640,3034],
      [176,13,54,54,40,7,168,0,0,19,19.9,19,19.9,1,0,26,3348,3348],
      [391,208,26,81,25,0,38,302,0,1.7,2,2.1,2.4,1,0,7,676,814],
      [263,65,45,101,42,10,176,87,0,3.5,3.5,3.5,3.5,1,0,3,911,911],
      [42,12,15,3,7,15,52,0,0,59.4,48,59.4,48,1,0,6.4,2496,2496],
      [190,98,20,16,63,5,160,42,0,6.2,5.8,6.2,5.8,1,0,10,1171,1171],
      [245,105,28,18,58,0,19,190,0,14.4,16.9,14.4,16.9,1,0,13,3532,3532],
      [77,28,0,42,0,0,0,70,0,5.7,6.2,6.8,7.5,1,0,1,436,525],
      [355,278,0,73,0,0,0,351,0,2.6,2.6,3.1,3.1,1,0,4,909,1095],
      [3156,2075,525,97,0,0,28,12,2657,2.9,3.4,3.5,4.1,1,0,6,9094,10957],
      [46,0,28,0,25,0,28,25,0,7.5,6.5,7.5,6.5,1,0,6,344,344],
      [56,14,12,15,7,5,53,0,0,5.3,5.6,5.3,5.6,4,0,3,296,296],
      [106,65,4,12,14,0,35,53,7,33,36.9,39.8,44.4,1,0,13,3503,4220],
      [71,31,28,9,7,0,31,23,21,3.5,3.3,4.2,3.9,1,0,1,246,296],
      [306,51,105,0,105,45,306,0,0,6.8,6.8,6.8,6.8,2,0,24,2082,2082],
      [244,68,78,22,62,14,244,0,0,0.8,0.8,0.8,0.8,4,0,7,191,191],
      [98,21,46,3,44,0,114,0,0,30.3,26.1,36.6,31.4,1,0,15,2974,3583],
      [331,100,44,61,107,0,312,0,0,1,1.1,1.2,1.3,1,0,2,328,395],
      [101,16,5,39,52,0,112,0,0,4,3.6,4,3.6,1,0,8,406,406],
      [192,9,102,0,10,77,198,0,0,9.3,9,11.2,10.9,1,0,8,1785,2151],
      [60,39,4,0,14,10,39,22,6,7.9,7,7.9,7,2,0,8,471,471],
      [180,74,18,33,57,0,153,29,0,3.9,3.8,3.9,3.8,2,0,5,700,700],
      [118,43,71,7,0,0,38,83,0,4.9,4.8,5.9,5.8,1,0,2,579,698],
      [73,43,10,4,15,0,6,66,0,16.6,16.8,20,20.3,1,0,3,1211,1459],
      [143,63,0,27,45,0,14,121,0,8,8.4,9.6,10.2,1,0,4,1139,1372],
      [2190,706,648,236,235,0,1825,0,0,6.6,8,7.6,9.1,2,0,19,14520,16690],
      [9,0,10,0,0,0,5,5,0,26.6,23.9,26.6,23.9,1,0,2,239,239],
      [203,21,167,3,10,0,201,0,0,6.2,6.3,7.5,7.6,1,0,5,1262,1520],
      [162,64,36,9,28,20,157,0,0,10.8,11.2,10.8,11.2,1,0,8,1754,1754],
      [183,60,21,29,71,0,181,0,0,8.3,8.4,10,10.1,1,0,6,1514,1824],
      [59,36,21,0,0,0,6,51,0,11.3,11.7,13.6,14.1,1,0,8,667,804],
      [412,198,14,99,91,10,412,0,0,14.2,14.2,14.2,14.2,1,0,20,5864,5864],
      [348,198,14,33,77,0,10,312,0,2.8,3,3.4,3.6,1,0,3,973,1172],
      [2529,554,513,66,703,220,2056,0,0,2.1,2.6,2.1,2.6,2,0,13,5333,5333],
      [250,28,92,57,0,30,164,43,0,7.1,8.6,7.1,8.6,2,0,8,1775,1775],
      [206,0,199,0,0,5,100,104,0,1,1,1.2,1.2,1,0,5,204,246],
      [617,184,136,103,133,0,377,179,0,5.1,5.7,6.1,6.8,1,0,7,3148,3793],
      [173,0,78,10,17,72,177,0,0,50.4,49.2,60.7,59.3,1,0,12,8716,10501],
      [286,45,156,27,37,0,265,0,0,5.1,5.5,5.1,5.5,1,0,6,1464,1464],
      [52,13,17,6,10,5,20,16,15,26.1,26.6,31.5,32.1,1,0,2,1358,1636],
      [45,6,28,0,0,25,59,0,0,6.2,4.8,6.2,4.8,2,0,7,281,281],
      [104,30,7,48,15,0,0,100,0,22.3,23.2,26.9,28,1,0,5,2321,2796],
      [465,234,44,104,118,0,431,15,54,8.7,8.1,10.5,9.8,1,0,13,4054,4884],
      [2145,862,63,491,518,300,2234,0,0,1.1,1.1,1.1,1.1,4,0,84,2400,2400],
      [283,20,125,39,91,17,292,0,0,20,19.3,20,19.3,1,0,13,5650,5650],
      [109,0,84,0,17,5,47,59,0,14,14.4,16.9,17.3,1,0,2,1525,1837],
      [160,0,21,72,25,40,74,68,16,3,3,3.6,3.6,1,0,2,477,575],
      [65,0,64,0,0,0,54,10,0,7.7,7.9,9.3,9.5,1,0,2,503,606],
      [366,160,27,96,72,0,260,84,11,23.1,23.9,27.9,28.7,1,0,10,8470,10205],
      [63,18,10,3,28,0,16,43,0,12.3,13.2,14.9,15.9,1,0,3,778,937],
      [309,126,13,59,153,30,381,0,0,8.4,6.8,8.4,6.8,4,0,7.5,2597,2597],
      [610,137,174,117,147,0,575,0,0,1.6,1.7,1.6,1.7,1,0,3,950,950],
      [562,165,125,93,119,0,502,0,0,10.2,11.4,10.2,11.4,3,0,13,5727,5727],
      [156,21,38,39,7,41,146,0,0,13.1,14,15.8,16.8,1,0,4,2040,2458],
      [75,30,5,19,21,0,40,35,0,49,49,59.1,59.1,1,0,8,3677,4430],
      [136,42,64,0,21,0,0,127,0,4.2,4.5,5.1,5.4,1,0,2,570,687],
      [96,13,56,0,10,40,119,0,0,12.7,10.3,47.2,38.1,2,0,8,1223,4530],
      [78,24,24,9,21,0,78,0,0,12.5,12.5,13.9,13.9,1,0,4,976,1084],
      [51,42,0,14,0,0,20,36,0,7.6,6.9,9.1,8.3,1,0,12,387,466],
      [1092,356,84,170,669,21,1300,0,0,4,3.4,4.9,4.1,1,0,13,4416,5320],
      [134,42,11,32,35,14,134,0,0,2.1,2.1,2.6,2.6,1,0,8,286,345],
      [75,24,9,0,36,0,0,69,0,4.1,4.4,4.9,5.3,1,0,5,305,367],
      [88,18,20,0,43,0,36,45,0,1.5,1.6,1.8,1.9,1,0,7,129,155],
      [3088,1061,1009,772,241,5,544,2544,0,1.4,1.4,1.4,1.4,1,0,8,4266,4266],
      [82,25,46,0,14,15,100,0,0,5.4,4.4,5.4,4.4,1,0,3,440,440],
      [110,30,66,0,7,0,18,85,0,12.4,13.2,12.4,13.2,1,0,2,1362,1362],
      [308,153,83,0,112,10,236,122,0,8.2,7.1,8.2,7.1,2,0,6,2533,2533],
      [143,75,32,3,62,0,172,0,0,1.6,1.3,1.6,1.3,1,0,10,225,225],
      [826,292,326,155,32,0,396,139,270,13.4,13.7,13.4,13.7,1,0,13,11045,11045],
      [3460,484,1831,318,208,142,2756,227,0,9.5,11,9.5,11,4,0,24,32760,32760],
      [73,15,32,0,28,5,80,0,0,28.1,25.7,33.9,30.9,1,0,8,2054,2475],
      [64,4,56,0,7,0,11,56,0,3.9,3.8,3.9,3.8,1,0,7,252,252],
      [497,122,28,113,273,10,454,92,0,4.8,4.3,4.8,4.3,2,0,8,2362,2362],
      [288,201,0,36,76,0,99,214,0,8.5,7.9,10.3,9.5,1,0,5,2459,2963],
      [494,134,62,140,91,67,494,0,0,17.6,17.6,17.6,17.6,1,0,21,8706,8706],
      [130,60,65,0,0,0,14,111,0,5.9,6.2,5.9,6.2,1,0,6,770,770],
      [204,47,43,6,29,104,229,0,0,39.8,35.4,39.8,35.4,1,0,36,8111,8111],
      [17,12,5,0,0,0,17,0,0,44.8,44.8,44.8,44.8,1,0,11,762,762],
      [60,10,25,0,31,5,71,0,0,2.3,1.9,2.7,2.3,1,0,1,136,164],
      [73,0,20,36,10,0,56,10,0,4.9,5.4,5.9,6.5,1,0,1,358,431],
      [169,32,14,25,28,70,169,0,0,2.8,2.8,3.4,3.4,1,0,6,481,580],
      [1351,379,449,271,219,20,1173,29,136,2.4,2.4,2.8,2.9,1,0,8,3189,3842],
      [2087,862,444,303,437,0,2046,0,0,10.8,11,10.8,11,4,0,9,22500,22500],
      [253,72,0,107,29,44,252,0,0,46.3,46.3,49.8,49.8,1,0,13,11719,12601],
      [102,37,26,10,29,0,102,0,0,1.8,1.8,2.4,2.4,1,0,5,183,244],
      [115,49,39,0,15,0,53,50,0,12.9,14.4,15.5,17.3,1,0,5,1482,1786],
      [288,18,186,21,15,60,300,0,0,4.3,4.2,4.3,4.2,2,0,11,1251,1251],
      [199,104,7,77,0,0,0,188,0,4.7,5,5.7,6,1,0,8,943,1136],
      [163,49,14,33,49,10,155,0,0,16.5,17.3,19.8,20.9,1,0,11,2684,3234],
      [97,21,41,0,49,15,16,100,10,7.5,5.8,7.5,5.8,2,0,9,729,729],
      [439,82,222,19,85,10,105,313,0,8.3,8.7,8.3,8.7,1,0,7,3630,3630],
      [3113,2019,609,248,157,80,3113,0,0,10.9,10.9,13.5,13.5,1,0,24,34085,42080],
      [153,35,7,57,24,20,143,0,0,18.7,20,21.5,23,2,0,3,2856,3283],
      [670,346,97,122,79,0,146,498,0,8.6,8.9,10.4,10.8,1,0,6,5757,6936],
      [129,19,93,3,10,30,0,155,0,4.9,4.1,4.9,4.1,2,0,15,631,631],
      [329,324,68,14,0,0,206,73,127,6.6,5.4,6.6,5.4,2,0,8,2184,2184],
      [5684,2221,454,820,1137,311,4943,0,0,3.7,4.3,3.7,4.3,1,0,37,21014,21014],
      [61,27,4,24,17,0,72,0,0,6.8,5.8,8.2,7,1,0,4,417,502],
      [267,27,133,51,34,89,70,178,86,10,8,10,8,2,0,9,2683,2683],
      [218,78,16,56,77,0,227,0,0,25.4,24.4,30.6,29.4,1,0,4,5532,6665],
      [919,458,236,56,58,76,884,0,0,5.2,5.4,6.3,6.6,1,0,6,4815,5801],
      [66,27,12,6,14,0,28,31,0,3.7,4.2,4.5,5,1,0,2,246,296],
      [54,36,0,16,0,0,0,52,0,12.7,13.2,15.3,15.9,1,0,1,686,827],
      [245,88,105,27,79,0,299,0,0,1.9,1.6,3.6,2.9,1,0,13,470,870],
      [54,24,28,0,0,0,0,52,0,7.3,7.6,8.8,9.2,1,0,2,395,476],
      [151,40,54,15,21,0,111,19,0,7.2,8.3,8.2,9.5,2,0,3,1080,1241],
      [103,45,4,27,29,0,105,0,0,13.6,13.3,13.6,13.3,2,0,7,1396,1396],
      [300,95,82,27,76,0,138,142,0,9.8,10.5,9.8,10.5,1,0,10,2933,2933],
      [249,126,4,98,21,0,58,191,0,18.3,18.3,18.3,18.3,1,0,9,4551,4551],
      [224,156,21,0,29,10,6,210,0,13.4,13.9,16.1,16.7,1,0,4,2999,3613],
      [1984,255,1008,192,158,0,1613,0,0,1.3,1.6,1.3,1.6,2,0,11,2540,2540],
      [58,15,37,0,0,5,0,57,0,0.4,0.5,0.5,0.5,1,0,2,26,31],
      [56,50,0,3,0,0,0,53,0,8,8.5,9.6,10.2,1,0,2,448,540],
      [244,91,12,44,60,50,257,0,0,3.7,3.5,4.5,4.2,1,0,3,906,1092],
      [12,0,0,6,7,0,10,3,0,17.6,16.2,17.6,16.2,2,0,13,211,211],
      [1416,402,432,0,252,65,376,775,0,1.2,1.5,1.2,1.5,2,0,12,1724,1724],
      [67,3,26,0,7,35,71,0,0,23.6,22.3,27.2,25.6,2,0,5,1584,1821],
      [95,36,0,32,24,0,0,92,0,1.6,1.6,1.9,1.9,1,0,7,148,178],
      [2376,638,233,639,580,131,2221,0,0,10.7,11.5,12.9,13.8,1,0,24,25482,30701],
      [442,225,102,57,91,0,82,393,0,1.5,1.4,1.5,1.4,1,0,5,683,683],
      [68,21,12,19,15,0,11,56,0,37.1,37.6,44.7,45.3,1,0,4,2521,3037],
      [268,76,96,45,41,10,248,20,0,10.9,10.9,10.9,10.9,2,0,9,2933,2933],
      [90,96,0,0,0,0,96,0,0,5.5,5.1,6.6,6.2,1,0,2,494,595],
      [72,23,25,7,17,0,55,17,0,6.8,6.8,6.8,6.8,1,0,7,486,486],
      [505,111,127,18,216,0,472,0,0,9.8,10.5,9.8,10.5,1,0,30,4955,4955],
      [450,112,145,59,105,0,356,58,7,13.6,14.6,13.6,14.6,1,0,7,6138,6138],
      [230,129,38,46,57,0,230,40,0,6.3,5.4,7.6,6.5,1,0,10,1451,1748],
      [242,97,78,4,43,0,0,222,0,1.3,1.4,1.5,1.7,1,0,2,311,375],
      [106,21,76,6,0,0,7,96,0,23.1,23.7,27.8,28.6,1,0,2,2445,2946],
      [31,10,14,0,7,0,4,27,0,7.6,7.6,7.6,7.6,1,0,6,237,237],
      [1186,422,306,109,168,230,1235,0,0,3.1,3,3.1,3,3,0,9,3711,3711],
      [256,69,198,0,0,5,272,0,0,11.5,10.8,11.5,10.8,2,0,19,2941,2941],
      [191,44,60,15,49,25,159,34,0,12.7,12.6,12.7,12.6,2,0,11,2430,2430],
      [213,102,63,6,28,0,186,13,0,11.9,12.7,14.3,15.3,1,0,10,2532,3051],
      [4562,1098,1278,498,1059,0,3933,0,0,5.8,6.7,5.8,6.7,1,0,15,26408,26408],
      [141,44,33,44,17,0,5,133,0,8.8,9,10.6,10.9,1,0,2,1244,1499],
      [187,23,65,17,91,5,201,0,0,3,2.8,3.6,3.4,1,0,6,562,677],
      [128,45,0,68,0,15,128,0,0,25.8,25.8,56.1,56.1,1,0,10,3303,7180],
      [194,70,16,0,45,47,46,132,0,2.6,2.8,3.6,3.9,1,0,12,499,703],
      [869,288,300,116,230,0,934,0,0,2.4,2.2,2.4,2.2,3,0,6,2078,2078],
      [125,51,9,33,22,0,20,84,11,18.6,20.2,22.4,24.4,1,0,3,2326,2802],
      [477,235,117,89,46,10,172,218,107,5.7,5.5,5.7,5.5,1,0,6,2741,2741],
      [717,716,13,10,24,0,51,712,0,4.9,4.6,6,5.6,1,0,5,3546,4272],
      [25,0,28,0,0,0,0,28,0,5.6,5,5.6,5,1,0,3,140,140],
      [73,9,61,3,0,0,73,0,0,19.2,19.2,19.2,19.2,1,0,8,1404,1404],
      [100,61,0,0,0,42,28,75,0,9.3,9,11.2,10.9,1,0,12,929,1119],
      [273,103,183,15,17,0,318,0,0,1.9,1.7,1.9,1.7,3,0,5,528,528],
      [587,108,119,94,203,0,524,0,0,6.4,7.2,6.4,7.2,3,0,15,3748,3748],
      [1437,920,154,41,203,0,1318,0,0,3.4,3.7,3.4,3.7,4,0,18,4900,4900],
      [97,3,101,3,0,7,108,6,0,19.1,16.3,19.1,16.3,2,0,12,1853,1853],
      [224,110,25,6,49,5,195,0,0,12.9,14.8,14.8,17,2,0,13,2890,3322],
      [321,168,46,9,77,15,315,0,0,3.4,3.4,3.4,3.4,4,0,11,1076,1076],
      [99,6,4,19,59,10,98,0,0,24.1,24.4,29.1,29.3,1,0,4,2387,2876],
      [416,179,40,54,109,7,318,65,6,3.8,4.1,4.6,4.9,1,0,6,1586,1911],
      [71,21,21,4,14,5,25,40,0,14,15.3,16.9,18.5,1,0,4,997,1201],
      [120,60,48,12,0,0,12,108,0,1.8,1.8,1.8,1.8,1,0,5,212,212],
      [213,72,43,3,84,30,132,97,3,6,5.5,6,5.5,2,0,12,1278,1278],
      [1634,625,387,223,399,0,1634,0,0,4.3,4.3,4.3,4.3,2,0,12,7060,7060],
      [275,184,11,48,7,0,7,243,0,2.6,2.8,2.6,2.8,1,0,9,702,702],
      [458,202,66,75,115,0,458,0,0,2,2,2,2,1,0,16,903,903],
      [218,28,118,44,0,10,91,109,0,31,33.8,37.4,40.8,1,0,8,6765,8151],
      [498,107,132,7,77,122,445,0,0,16.5,18.5,16.5,18.5,3,0,19,8227,8227],
      [143,54,65,0,14,0,11,122,0,2.3,2.5,2.8,3,1,0,4,334,402],
      [722,163,0,202,237,20,622,0,0,5.8,6.7,5.8,6.7,1,0,9,4164,4164],
      [236,27,192,12,37,0,268,0,0,29,25.5,29,25.5,2,0,15,6844,6844],
      [239,39,70,18,62,55,244,0,0,6.9,6.8,6.9,6.8,2,0,10,1661,1661],
      [1882,711,589,218,442,0,1960,0,0,3.2,3.1,3.2,3.1,2,0,10,6068,6068],
      [291,188,29,0,69,37,323,0,0,20.8,18.8,22.4,20.2,2,0,14,6063,6519],
      [328,124,71,61,66,0,322,0,0,2,2.1,2,2.1,1,0,13,665,665],
      [264,126,7,50,84,0,267,0,0,2.1,2,2.1,2,1,0,6,544,544],
      [270,54,42,53,91,30,270,0,0,3.3,3.3,3.3,3.3,1,0,6,893,893],
      [32,0,7,12,0,10,29,0,0,11,12.1,11,12.1,1,0,2.5,352,352],
      [1382,353,245,7,581,253,1439,0,0,2.4,2.3,2.4,2.3,4,0,8,3344,3344],
      [98,45,5,19,28,0,94,3,0,4.1,4.2,5,5,1,0,1,404,487],
      [289,3,100,0,66,145,314,0,0,19.8,18.3,19.8,18.3,2,0,8,5732,5732],
      [1059,299,234,144,217,60,954,0,0,21.6,24,24.9,27.6,2,0,17,22920,26345],
      [724,331,12,189,164,0,530,142,24,13,13.5,15.7,16.3,1,0,9,9409,11336],
      [40,10,22,0,17,0,49,0,0,7.1,5.8,7.1,5.8,4,0,4,285,285],
      [128,72,18,24,7,0,4,117,0,10.1,10.7,12.2,12.9,1,0,3,1293,1558],
      [1248,249,262,146,423,15,1095,0,0,12.2,13.8,12.2,13.8,4,0,12,15165,15165],
      [387,109,118,67,90,7,391,0,0,8.5,8.4,10.2,10.1,1,0,17,3287,3960],
      [73,0,65,0,0,5,19,44,7,10.5,10.9,12.6,13.1,1,0,3,764,920],
      [151,22,0,68,47,0,32,105,0,14.5,16,17.5,19.3,1,0,3,2194,2643],
      [465,48,162,0,196,37,443,0,0,4.3,4.5,5.2,5.5,1,0,4,2014,2427],
      [63,29,0,9,24,0,62,0,0,13.1,13.3,15.8,16.1,1,0,4,827,996],
      [226,88,70,19,28,0,92,106,7,11.1,12.2,11.1,12.2,1,0,10,2508,2508],
      [115,76,11,28,0,0,10,105,0,1.7,1.7,1.7,1.7,1,0,1.5,201,201],
      [189,39,112,3,28,0,0,182,0,3,3.1,3,3.1,2,0,15,563,563],
      [55,52,0,6,0,0,58,0,0,9.7,9.2,11.7,11.1,1,0,11,534,643],
      [85,12,44,0,24,0,12,68,0,12.2,13,14.7,15.7,1,0,4,1039,1252],
      [2143,219,657,3,485,437,1801,0,0,20.2,24,24.3,29,1,0,7,43303,52172],
      [520,181,160,9,55,25,430,0,0,6,7.3,6,7.3,2,0,8,3133,3133],
      [113,25,34,16,27,0,38,64,0,15.9,17.7,18.3,20.3,2,0,3,1801,2070],
      [264,85,17,26,133,0,157,98,6,10,10.1,12,12.2,1,0,6,2639,3180],
      [349,162,24,18,112,30,342,4,0,6.8,6.8,6.8,6.8,2,0,4,2360,2360],
      [229,88,5,0,17,131,241,0,0,11,10.5,12.7,12,2,0,5,2520,2897],
      [172,88,40,4,30,7,169,0,0,10.4,10.6,10.4,10.6,4,0,5,1788,1788],
      [323,108,57,53,74,31,323,0,0,3.8,3.8,3.8,3.8,1,0,13,1238,1238],
      [89,15,28,17,7,30,24,73,0,5,4.6,5,4.6,2,0,4,442,442],
      [878,113,404,161,133,17,828,0,0,3.8,4.1,4.3,4.5,1,0,5,3368,3742],
      [220,81,50,51,42,0,224,0,0,1.3,1.3,1.3,1.3,3,0,3,281,281],
      [150,125,4,20,0,0,6,143,0,4.6,4.6,5.5,5.6,1,0,3,687,828],
      [161,60,38,30,41,0,169,0,0,1.4,1.3,1.4,1.3,1,0,2,220,220],
      [804,345,334,0,236,42,957,0,0,14.2,11.9,17.1,14.3,1,0,12,11388,13720],
      [272,63,20,61,71,30,245,0,0,35.7,39.7,41.1,45.6,2,0,14,9720,11172],
      [33,8,7,0,10,15,32,8,0,4.4,3.6,4.4,3.6,2,0,2,145,145],
      [64,20,28,10,7,0,65,0,0,2.2,2.2,2.2,2.2,2,0,3,140,140],
      [277,48,110,47,49,0,84,170,0,4.6,5,4.6,5,2,0,19,1265,1265],
      [322,78,100,67,54,0,179,106,14,16.3,17.5,19.6,21.1,1,0,13,5242,6316],
      [164,52,56,18,7,20,110,43,0,17,18.2,17,18.2,2,0,11,2791,2791],
      [145,48,10,72,14,0,144,0,0,5.1,5.2,6.2,6.2,1,0,3,742,894],
      [73,36,24,12,0,0,0,72,0,8,8.1,9.6,9.8,1,0,5,584,704],
      [245,42,179,24,42,15,302,0,0,3.6,3,3.6,3,2,0,7,893,893],
      [406,209,36,125,20,0,102,288,0,13.9,14.5,16.7,17.4,1,0,7,5639,6794],
      [98,35,0,41,20,0,28,68,0,56.5,57.6,68,69.4,1,0,5,5533,6666],
      [83,44,10,3,25,0,15,63,4,4.9,5,6,6,1,0,3,410,494],
      [182,30,59,39,29,0,0,157,0,5.5,6.4,6.6,7.7,1,0,5,1001,1206],
      [820,460,98,199,175,0,685,212,35,3.1,2.7,3.1,2.7,4,0,11,2552,2552],
      [545,116,226,131,72,0,79,466,0,16,16,16,16,1,0,11,8740,8740],
      [105,96,9,0,0,0,21,84,0,1.9,1.9,1.9,1.9,1,0,8,198,198],
      [55,12,10,14,15,0,9,42,0,42.7,46,51.4,55.5,1,0,1,2347,2828],
      [62,27,4,0,31,7,69,0,0,13.2,11.8,15.9,14.3,1,0,8,817,984],
      [109,0,105,0,0,0,0,105,0,2.3,2.4,2.8,2.9,1,0,1,249,300],
      [436,27,334,15,0,60,436,0,0,17.8,17.8,17.8,17.8,2,0,15,7762,7762],
      [179,60,44,6,38,40,188,0,0,4.3,4.1,5.2,5,1,0,4,776,935],
      [1710,932,521,233,214,0,1900,0,0,7.7,6.9,7.7,6.9,2,0,21,13130,13130],
      [128,49,15,25,31,0,120,0,0,12.8,13.7,14.8,15.8,2,0,2,1644,1890],
      [174,21,7,30,50,32,140,0,0,77.7,96.6,77.7,96.6,4,0,8,13528,13528],
      [70,28,4,15,21,0,39,29,0,8.8,9.1,10.6,10.9,1,0,5,617,743],
      [379,159,29,57,119,15,379,0,0,6.3,6.3,6.3,6.3,1,0,7,2391,2391],
      [258,12,16,230,0,0,26,232,0,2.3,2.3,2.3,2.3,2,0,5,605,605],
      [129,33,0,39,24,25,121,0,0,16,17.1,18.4,19.6,2,0,4,2064,2372],
      [856,349,194,74,84,155,856,0,0,3,3,3,3,2,0,12,2551,2551],
      [127,56,14,10,39,0,25,94,0,13.1,14,15.8,16.9,1,0,7,1668,2010],
      [381,119,107,9,28,65,137,191,0,4.8,5.5,4.8,5.5,2,0,19,1817,1817],
      [234,117,19,93,0,5,231,3,0,21.8,21.8,21.8,21.8,1,0,10,5103,5103],
      [215,58,62,46,51,0,82,135,0,13.2,13.1,15.9,15.7,1,0,8,2836,3417],
      [1406,541,70,290,335,90,1326,0,0,10.6,11.3,10.5,11.1,4,0,1,14938,14698],
      [85,21,67,3,0,15,22,59,25,12.3,9.9,12.3,9.9,2,0,13,1049,1049],
      [59,10,32,3,21,5,71,0,0,18,14.9,21.7,18,1,0,10,1061,1278],
      [358,139,0,89,70,40,338,0,0,38.9,41.2,46.8,49.6,1,0,9,13919,16770],
      [73,40,0,12,10,0,0,62,0,50.3,59.3,60.7,71.4,1,0,7,3675,4428],
      [597,189,96,78,239,7,609,0,0,4.3,4.2,4.3,4.2,1,0,6,2542,2542],
      [276,90,120,34,14,0,101,139,18,8.1,8.7,8.1,8.7,1,0,2,2244,2244],
      [728,181,165,15,84,167,330,27,255,11.1,13.1,13.3,15.8,1,0,4,8046,9694],
      [59,0,55,0,0,0,11,11,33,83.8,89.9,101,108.3,1,0,6,4945,5958],
      [98,36,32,9,17,0,45,42,7,13.4,14,16.1,16.8,1,0,11,1313,1582],
      [756,452,66,127,84,5,302,320,112,4.3,4.4,4.3,4.4,1,0,7,3214,3214],
      [155,120,34,0,0,0,47,107,0,3.1,3.1,3.8,3.8,1,0,10,483,582],
      [544,125,104,107,133,0,107,362,0,4.2,4.9,5.1,5.9,1,0,4,2306,2778],
      [813,27,677,34,0,100,18,820,0,2,1.9,2,1.9,2,0,9,1634,1634],
      [75,23,0,30,27,14,94,0,0,9.6,7.7,9.6,7.7,1,0,6,720,720],
      [112,85,4,16,14,0,119,0,0,9.6,9,9.6,9,4,0,13,1073,1073],
      [211,73,32,79,22,5,111,100,0,16.9,16.9,16.9,16.9,1,0,13,3566,3566],
      [895,222,271,222,63,0,778,0,0,0.9,1,0.9,1,1,0,7,780,780],
      [74,18,7,37,7,5,74,0,0,54,54,54,54,1,0,14,3995,3995],
      [3331,1049,519,399,595,360,2922,0,0,14.7,16.8,14.7,16.8,4,0,42,49034,49034],
      [64,15,18,9,15,5,12,39,11,6.1,6.3,7.4,7.6,1,0,6,392,472],
      [1402,328,804,187,49,63,1431,0,0,12.6,12.3,12.3,12,1,0,23,17607,17219],
      [337,128,21,86,77,25,337,0,0,1.8,1.8,1.9,1.9,1,0,8,591,657],
      [73,0,0,67,0,10,39,38,0,11.1,10.5,13.4,12.7,1,0,6,812,978],
      [64,12,28,17,7,0,47,17,0,8.8,8.8,10.6,10.6,1,0,12,564,680],
      [1703,186,463,0,516,316,1481,0,0,5,5.8,5,5.8,3,0,8,8549,8549],
      [151,41,21,29,41,19,151,0,0,2.6,2.6,2.9,2.9,1,0,3,391,434],
      [66,27,14,6,17,0,42,22,0,27.5,28.4,33.2,34.2,1,0,2,1817,2189],
      [70,24,0,3,42,5,74,0,0,7.7,7.3,9.4,8.9,2,0,3,540,659],
      [754,301,130,204,90,0,213,512,0,9.9,10.3,11.9,12.4,1,0,8,7443,8967],
      [325,106,49,31,80,12,136,142,0,26.7,31.2,32.1,37.5,1,0,6,8663,10437],
      [168,84,7,45,40,22,198,0,0,27.5,23.3,27.5,23.3,2,0,12,4622,4622],
      [226,99,50,43,63,5,260,0,0,4,3.5,4,3.5,4,0,7,915,915],
      [335,83,71,71,111,17,353,0,0,75.8,72,75.8,72,4,0,12,25401,25401],
      [76,21,51,3,0,0,63,12,0,7.5,7.6,9,9.1,1,0,5,569,686],
      [221,31,133,0,45,5,53,161,0,6.8,7.1,8.2,8.5,1,0,4,1512,1822],
      [1010,213,224,111,212,75,758,77,0,5,6.1,5,6.1,2,0,11,5053,5053],
      [116,46,12,22,35,15,130,0,0,16.7,14.9,16.7,14.9,1,0,10,1939,1939],
      [39,25,0,3,14,0,11,31,0,2.9,2.7,2.9,2.7,2,0,1,114,114],
      [170,75,12,23,39,0,108,41,0,9,10.3,9,10.3,4,0,8,1532,1532],
      [179,85,0,49,59,0,193,0,0,26.4,24.5,31.8,29.5,1,0,4,4723,5690],
      [166,126,0,38,0,0,0,164,0,17.9,18.1,21.6,21.8,1,0,13,2971,3580],
      [354,48,34,174,73,25,354,0,0,16.5,16.5,16.5,16.5,1,0,10,5841,5841],
      [615,546,0,74,7,0,627,0,0,1.5,1.5,1.5,1.5,2,0,8,941,941],
      [126,46,7,18,49,15,135,0,0,8.2,7.7,8.2,7.7,2,0,8,1035,1035],
      [1390,379,377,133,182,170,1241,0,0,4,4.5,4,4.5,3,0,13,5597,5597],
      [129,30,59,10,39,0,94,44,0,23.2,21.7,27.9,26.1,1,0,8,2988,3600],
      [203,35,48,6,14,100,203,0,0,15.9,15.9,15.9,15.9,1,0,5,3219,3219],
      [758,116,312,132,30,58,648,0,0,6.1,7.1,6.1,7.1,1,0,15,4630,4630],
      [558,92,116,57,196,0,461,0,0,7.4,8.9,7.4,8.9,1,0,6,4112,4112],
      [25,14,0,4,7,0,22,0,3,23.2,23.2,23.2,23.2,1,0,6,580,580],
      [850,94,348,205,45,74,320,446,0,12.9,14.3,15.5,17.2,1,0,5,10942,13183],
      [235,78,10,123,68,0,13,82,184,4,3.4,4.8,4,1,0,9,937,1129],
      [58,15,0,19,22,0,26,30,0,43.4,44.9,52.2,54.1,1,0,8,2515,3030],
      [316,14,118,14,155,0,139,151,11,8.2,8.6,8.2,8.6,2,0,8,2600,2600],
      [7633,1240,2455,135,1732,1572,1376,5193,565,3.3,3.5,3.3,3.5,2,0,19,25217,25217],
      [119,0,118,0,0,0,95,19,4,1.5,1.5,1.8,1.8,1,0,2,176,212],
      [55,12,17,6,14,10,59,0,0,6.2,5.8,7.5,7,1,0,3,341,411],
      [1644,240,858,177,91,115,1481,0,0,9.9,11,9.9,11,1,0,38,16357,16357],
      [1193,196,667,149,115,54,516,485,180,2.5,2.5,3,3.1,1,0,12,3006,3622],
      [605,155,231,91,105,0,582,0,0,2.8,2.9,3.4,3.5,1,0,8,1694,2041],
      [389,120,94,77,63,10,198,166,0,2.9,3.1,2.9,3.1,2,0,6,1134,1134],
      [132,56,0,27,38,7,128,0,0,9,9.3,10.8,11.2,1,0,8,1186,1429],
      [132,30,50,22,24,0,39,27,60,10.5,11,12.6,13.2,1,0,9,1383,1666],
      [286,94,36,44,38,55,267,0,0,6,6.5,6,6.5,1,0,6,1727,1727],
      [305,73,87,39,63,10,272,0,0,14.8,16.6,14.8,16.6,3,0,7,4504,4504],
      [268,73,105,0,70,0,248,0,0,17.9,19.4,20.6,22.2,2,0,8,4800,5517],
      [110,36,15,0,49,10,110,0,0,23.2,23.2,23.2,23.2,2,0,24,2553,2553],
      [3963,1184,616,952,1091,42,3885,0,0,4.3,4.4,4.3,4.4,2,0,11,16921,16921],
      [397,83,252,16,58,0,409,0,0,18.8,18.2,22.6,22,1,0,7,7459,8987],
      [254,90,39,3,117,0,119,100,30,15.3,15.6,15.3,15.6,2,0,11,3877,3877],
      [134,73,0,22,14,0,33,76,0,15.6,19.2,18.8,23.1,1,0,8,2088,2516],
      [266,101,70,26,56,5,258,0,0,17.7,18.2,21.3,21.9,1,0,8,4695,5657],
      [237,114,11,44,42,0,181,30,0,13.5,15.2,16.3,18.3,1,0,9,3201,3857],
      [1243,555,15,245,394,10,1219,0,0,9.2,9.4,11.1,11.4,1,0,9,11490,13843],
      [666,138,233,124,105,5,605,0,0,1.1,1.2,1.1,1.2,1,0,13,752,752],
      [1810,349,423,377,457,10,1616,0,0,16.2,18.2,16.2,18.2,3,0,35,29399,29399],
      [88,56,7,15,7,0,9,76,0,26.1,27,31.5,32.6,1,0,6,2299,2770],
      [62,19,8,9,15,0,51,0,0,14.2,17.3,15.3,18.6,1,0,5,883,949],
      [305,59,102,19,60,21,138,123,0,28.5,33.3,34.4,40.2,1,0,10,8702,10484],
      [349,107,93,97,52,0,286,63,0,10.6,10.6,10.6,10.6,1,0,8,3703,3703],
      [156,0,133,0,10,0,143,0,0,4.5,4.9,5.4,5.9,1,0,4,705,849],
      [100,24,76,3,7,0,40,70,0,14.3,13,14.3,13,1,0,15,1434,1434],
      [222,20,5,93,73,0,47,144,0,11,12.8,13.3,15.5,1,0,4,2450,2952],
      [506,54,161,43,173,80,511,0,0,6.9,6.8,6.9,6.8,1,0,9,3500,3500],
      [123,28,0,47,15,25,115,0,0,15.4,16.5,17.7,18.9,2,0,3,1896,2179],
      [252,0,109,143,0,0,109,143,0,27.1,27.1,27.1,27.1,2,0,11,6821,6821],
      [258,186,40,19,10,0,0,255,0,5.4,5.5,6.5,6.6,1,0,4,1393,1678],
      [166,67,88,6,0,0,37,120,4,15.7,16.2,18.9,19.5,1,0,7,2604,3137],
      [181,45,39,38,42,5,169,0,0,20.1,21.5,22.6,24.2,2,0,10,3638,4088],
      [284,99,112,26,28,0,220,27,18,11.4,12.2,11.4,12.2,1,0,4,3234,3234],
      [52,0,60,0,0,0,0,55,5,4.8,4.2,5.8,5.1,1,0,3,252,304],
      [102,72,0,18,0,7,0,97,0,10.1,10.6,12.1,12.8,1,0,3,1028,1239],
      [58,26,29,3,0,0,22,36,0,11.9,11.9,14.4,14.4,1,0,5,691,833],
      [1196,626,108,56,413,5,1208,0,0,6.5,6.5,6.5,6.5,4,0,15,7824,7824],
      [38,30,0,8,0,0,32,0,6,6.5,6.5,6.5,6.5,1,0,4,247,247],
      [84,33,0,21,27,0,61,20,0,32.6,33.8,39.3,40.7,1,0,10,2738,3299],
      [73,18,14,8,44,0,40,44,0,15.4,13.3,18.5,16.1,1,0,6,1121,1351],
      [449,352,12,12,52,0,0,428,0,35.7,37.5,43,45.2,1,0,8,16042,19328],
      [551,188,116,52,114,5,223,231,21,4.4,5.1,5.3,6.2,1,0,13,2435,2934],
      [194,48,47,28,27,35,185,0,0,7.4,7.8,7.4,7.8,2,0,8,1440,1440],
      [185,36,14,50,32,10,142,0,0,34.7,45.2,34.7,45.2,1,0,16,6416,6416],
      [134,43,0,77,36,0,156,0,0,9.1,7.8,10.9,9.4,1,0,5,1213,1461],
      [1241,575,45,319,301,0,1240,0,0,8.2,8.2,8.2,8.2,1,0,15,10222,10222],
      [99,81,0,3,15,0,6,93,0,2.4,2.4,2.4,2.4,1,0,3,241,241],
      [1285,685,67,133,620,7,1512,0,0,0.9,0.8,0.9,0.8,1,0,16.5,1200,1200],
      [466,164,212,12,66,7,461,0,0,3.7,3.7,4.4,4.5,1,0,10,1708,2058],
      [77,30,0,20,22,0,7,65,0,27.4,29.3,33,35.3,1,0,4,2108,2540],
      [81,11,22,0,20,21,64,10,0,5.6,6.1,6.8,7.4,1,0,1,454,547],
      [694,61,164,168,145,60,598,0,0,5.9,6.8,5.9,6.8,4,0,8,4086,4086],
      [62,18,7,6,10,17,7,51,0,32.7,34.9,39.4,42.1,1,0,5,2027,2442],
      [56,0,42,0,0,15,57,0,0,5.2,5.1,5.2,5.1,1,0,3,293,293],
      [753,289,13,0,363,7,672,0,0,5.7,6.4,5.7,6.4,1,0,12,4301,4301],
      [421,177,22,217,0,5,421,0,0,39.9,39.9,39.9,39.9,1,0,15.5,16788,16788],
      [59,30,11,3,21,0,65,0,0,9.6,8.7,11.6,10.5,1,0,3,566,682],
      [409,42,201,39,67,60,409,0,0,6.5,6.5,6.5,6.5,2,0,11,2672,2672],
      [115,21,90,4,0,0,94,21,0,16.3,16.3,16.3,16.3,2,0,12,1869,1869],
      [93,58,20,4,10,0,22,70,0,8.6,8.7,10.4,10.5,1,0,3,803,967],
      [71,30,15,12,7,5,0,69,0,56.6,58.2,68.2,70.2,1,0,7,4019,4842],
      [971,345,42,249,150,10,796,0,0,5,6.1,6,7.4,1,0,14,4867,5864],
      [74,15,7,42,0,10,74,0,0,42.3,42.3,42.3,42.3,1,0,17,3132,3132],
      [81,18,4,25,21,10,25,30,23,1.1,1.1,1.3,1.4,1,0,1,89,107],
      [106,6,33,19,0,55,113,0,0,10.6,10,12.8,12,1,0,7,1126,1357],
      [1168,516,240,176,314,10,1256,0,0,1.3,1.2,1.3,1.2,3,0,8,1538,1538],
      [1093,392,119,235,166,47,959,0,0,6.6,7.6,6.6,7.6,1,0,11,7263,7263],
      [67,10,18,31,0,0,0,59,0,57.5,65.3,69.3,78.7,1,0,2,3853,4642],
      [1256,270,121,450,90,50,981,0,0,23,29.4,41,52.5,1,0,28,28855,51527],
      [72,67,5,0,0,0,6,66,0,17.3,17.3,20.9,20.9,1,0,15,1248,1504],
      [777,287,159,38,147,45,676,0,0,19.4,22.2,22.2,25.6,2,0,13,15039,17286],
      [131,30,33,18,35,12,128,0,0,37.8,38.6,45.5,46.6,1,0,13,4947,5960],
      [134,78,0,0,63,0,141,0,0,15.1,14.3,18.2,17.3,1,0,8,2019,2433],
      [826,255,295,72,157,0,779,0,0,6.5,6.9,6.5,6.9,1,0,42,5400,5400],
      [3180,916,538,295,969,0,2718,0,0,10.4,12.2,10.4,12.2,4,0,36,33028,33028],
      [324,175,108,29,17,5,334,0,0,15.7,15.2,15.7,15.2,1,0,13,5078,5078],
      [220,132,45,30,7,0,49,165,0,3.2,3.3,3.2,3.3,1,0,3,701,701],
      [258,34,89,6,63,49,241,0,0,2.9,3.1,2.9,3.1,1,0,12,736,736],
      [139,25,19,6,29,60,139,0,0,7,7,7,7,2,0,7,976,976],
      [274,70,123,57,28,30,308,0,0,4.3,3.8,4.3,3.8,2,0,4,1167,1167],
      [231,0,86,0,50,130,266,0,0,16.3,14.1,16.3,14.1,4,0,12,3762,3762],
      [126,58,15,6,21,40,22,118,0,4.9,4.4,4.9,4.4,2,0,5,619,619],
      [93,0,70,0,15,10,0,95,0,11.8,11.6,14.3,14,1,0,8,1101,1327],
      [1825,594,569,192,259,60,1674,0,0,0.7,0.7,0.7,0.7,1,0,15,1210,1210],
      [500,229,31,130,60,0,170,227,53,11.8,13.1,14.2,15.7,1,0,8,5876,7080],
      [373,129,96,21,105,50,389,12,0,6.4,6,6.4,6,2,0,10,2395,2395],
      [330,179,26,64,110,0,379,0,0,3.2,2.8,3.2,2.8,1,0,10,1050,1050],
      [57,27,28,0,0,0,0,55,0,30.5,31.7,36.8,38.1,1,0,4,1741,2098],
      [127,61,0,20,44,0,13,98,14,12.4,12.6,15,15.2,1,0,4,1579,1902],
      [1362,579,291,171,378,0,1419,0,0,2.3,2.3,2.3,2.3,3,0,11,3193,3193],
      [1463,780,119,223,371,0,1493,0,0,24.6,24.1,24.6,24.1,4,0,23,36046,36046],
      [148,12,49,36,28,15,140,0,0,23.2,24.6,28,29.6,1,0,3,3437,4141],
      [474,71,24,343,21,15,41,433,0,3.9,3.9,3.9,3.9,1,0,6,1829,1829],
      [77,24,18,21,14,0,18,59,0,5.3,5.3,6.4,6.4,1,0,10,408,492],
      [213,105,67,41,0,0,52,161,0,3.8,3.8,3.8,3.8,1,0,2.5,801,801],
      [175,22,53,0,100,0,59,116,0,3.4,3.4,3.4,3.4,2,0,7,597,597],
      [360,122,98,54,62,0,336,0,0,2.1,2.3,2.1,2.3,1,0,18,763,763],
      [331,84,0,12,0,284,156,96,128,3.7,3.3,3.7,3.3,2,0,5,1238,1238],
      [285,48,187,15,35,25,310,0,0,4.4,4,4.4,4,2,0,7,1249,1249],
      [187,72,32,10,44,5,4,159,0,73.5,84.4,84.5,97,2,0,3,13752,15807],
      [502,120,32,89,83,120,444,0,0,18.5,20.9,18.5,20.9,1,0,14,9296,9296],
      [303,0,266,0,0,25,0,291,0,1.1,1.1,1.3,1.4,1,0,1,329,396],
      [174,105,83,0,0,5,98,35,60,17.7,16,17.7,16,2,0,17,3088,3088],
      [44,14,0,16,17,5,49,3,0,8.6,7.3,8.6,7.3,4,0,2,380,380],
      [387,88,221,37,34,0,18,362,0,6,6.1,7.3,7.4,1,0,7,2333,2811],
      [406,18,374,27,0,0,21,398,0,5.2,5,5.2,5,2,0,19,2109,2109],
      [496,198,109,67,90,60,524,0,0,20.3,19.2,23.4,22.1,2,0,14,10080,11586],
      [116,34,38,4,30,0,33,73,0,4.8,5.2,4.8,5.2,1,0,2,553,553],
      [53,17,17,0,24,5,63,0,0,6.7,5.6,6.7,5.6,2,0,16,354,354],
      [209,42,145,10,43,0,225,15,0,0.9,0.8,0.9,0.8,1,0,2.5,190,190],
      [903,336,134,158,148,100,387,303,186,6.1,6.3,7.3,7.5,1,0,11,5482,6605],
      [469,71,170,15,67,96,419,0,0,6.8,7.6,8.2,9.2,1,0,11,3191,3845],
      [1894,187,1241,143,156,10,966,696,75,2.1,2.3,2.1,2.3,2,0,12,3986,3986],
      [215,108,0,94,7,0,76,133,0,2.6,2.7,2.6,2.7,1,0,7,560,560],
      [82,38,9,20,14,0,29,52,0,1,1,1.2,1.2,1,0,2,80,96],
      [1694,391,469,385,210,85,1540,0,0,4.6,5.1,4.6,5.1,1,0,16,7816,7816],
      [81,24,22,20,10,0,64,4,8,24.7,26.3,29.8,31.7,1,0,5,2002,2412],
      [297,150,7,132,0,0,13,264,12,5.2,5.4,5.2,5.4,1,0,6,1557,1557],
      [280,161,91,0,10,0,262,0,0,5.1,5.4,5.1,5.4,4,0,7,1417,1417],
      [56,19,5,16,10,0,29,21,0,22.3,24.9,26.8,30,1,0,5,1246,1501],
      [492,114,97,157,49,75,248,244,0,8.5,8.5,8.5,8.5,1,0,11,4158,4158],
      [426,127,122,54,87,20,31,352,27,1.5,1.6,1.8,1.9,1,0,2,646,778],
      [499,262,42,60,90,0,155,299,0,4.6,5.1,4.6,5.1,1,0,10,2310,2310],
      [60,9,32,12,7,0,60,0,0,16.5,16.5,18.4,18.4,1,0,5,992,1102],
      [362,152,68,41,72,15,348,0,0,19.6,20.4,23.6,24.5,1,0,6,7091,8543],
      [363,169,23,115,42,0,82,267,0,9.5,9.8,11.4,11.8,1,0,5,3431,4134],
      [53,24,0,24,0,0,0,48,0,19,21,22.9,25.3,1,0,3,1007,1213],
      [17518,9404,1221,0,2955,0,13580,0,0,3.1,4,3.1,4,1,0,14,54620,54620],
      [199,82,92,13,10,0,20,100,77,4.4,4.4,5.3,5.3,1,0,2,871,1049],
      [2171,534,319,323,87,0,209,984,70,9.1,15.6,9.1,15.6,1,0,8,19699,19699],
      [220,90,19,64,39,0,12,200,0,8.8,9.1,10.6,11,1,0,4,1930,2325],
      [1355,535,428,58,224,10,1255,0,0,5.5,6,5.5,6,1,0,26,7505,7505],
      [374,170,4,88,126,10,398,0,0,0.8,0.8,1,0.9,1,0,2,300,363],
      [62,51,0,0,10,5,66,0,0,5.6,5.3,6.8,6.4,1,0,3,349,420],
      [311,11,217,3,36,57,324,0,0,6.6,6.3,7.9,7.6,1,0,6,2038,2455],
      [132,32,15,13,31,45,136,0,0,4.6,4.4,5.5,5.3,1,0,6,601,724],
      [135,60,17,0,70,0,0,147,0,4.9,4.5,4.9,4.5,2,0,15,667,667],
      [79,16,34,0,28,10,81,7,0,45.4,40.7,54.7,49.1,1,0,5,3585,4319],
      [155,75,28,10,29,0,14,128,0,12.4,13.5,14.9,16.3,1,0,4,1916,2308],
      [965,433,234,148,245,0,1060,0,0,3.8,3.4,4.1,3.7,1,0,30,3655,3930],
      [296,27,166,48,21,50,312,0,0,18.4,17.4,18.4,17.4,2,0,10,5432,5432],
      [56,3,19,10,24,0,56,0,0,3,3,3,3,2,0,3,170,170],
      [206,88,23,31,35,0,73,104,0,8.8,10.3,10.7,12.4,1,0,8,1823,2196],
      [651,104,385,16,49,27,581,0,0,5.5,6.2,6.7,7.5,1,0,4,3612,4352],
      [2323,804,332,0,947,240,636,1552,135,1.7,1.7,1.7,1.7,2,0,6,3981,3981],
      [146,46,0,50,41,0,44,93,0,13.6,14.5,16.4,17.4,1,0,6,1984,2390],
      [284,49,83,75,51,5,80,183,0,12.8,13.9,15.5,16.7,1,0,5,3646,4393],
      [249,47,26,80,84,0,237,0,0,15.3,16,15.3,16,2,0,9,3800,3800],
      [32,4,7,0,0,30,41,0,0,15.6,12.1,15.6,12.1,1,0,3,498,498],
      [1908,401,1216,155,107,10,1889,0,0,14.2,14.3,14.2,14.3,3,0,48,27000,27000],
      [253,56,149,0,24,24,76,177,0,6.7,6.7,6.7,6.7,1,0,7,1694,1694],
      [184,53,21,43,38,15,170,0,0,16,17.4,18.4,20,2,0,4,2952,3393],
      [76,60,12,0,0,0,0,72,0,4.9,5.2,5.9,6.3,1,0,1,374,451],
      [276,108,5,75,56,0,177,53,14,1.1,1.2,1.1,1.2,1,0,4,300,300],
      [3471,1327,257,534,995,42,3155,0,0,2.2,2.4,2.2,2.4,4,0,17,7575,7575],
      [380,67,103,77,98,20,365,0,0,8.7,9.1,10.5,11,1,0,9,3322,4002],
      [155,18,0,69,14,47,148,0,0,33.5,35.1,33.5,35.1,1,0,8,5196,5196],
      [327,64,142,21,0,100,327,0,0,7.8,7.8,7.8,7.8,2,0,16,2566,2566],
      [193,82,80,50,14,0,41,182,3,8.7,7.5,10.5,9,1,0,5,1686,2031],
      [1108,279,20,314,273,30,916,0,0,22.6,27.4,24.3,29.4,1,0,6,25054,26940],
      [363,168,10,92,77,5,0,352,0,0.3,0.3,0.4,0.4,1,0,10,117,141],
      [249,144,104,6,14,0,268,0,0,2,1.9,2,1.9,1,0,3,500,500],
      [769,265,195,235,52,0,178,557,12,16.3,16.8,16.3,16.8,1,0,8,12551,12551],
      [137,54,42,30,7,0,97,36,0,2.9,3,2.9,3,1,0,3,399,399],
      [579,215,81,109,112,0,517,0,0,1.4,1.5,1.4,1.5,1,0,7,784,784],
      [133,39,0,41,24,20,124,0,0,15.9,17,18.3,19.6,2,0,6,2112,2428],
      [78,7,34,6,21,0,22,46,0,7.7,8.8,8.8,10.1,2,0,3,600,690],
      [101,12,31,3,50,0,65,28,3,4.5,4.7,4.5,4.7,1,0,1.5,450,450],
      [1222,503,333,173,150,5,1164,0,0,4.7,5,4.7,5,1,0,20,5800,5800],
      [1093,486,277,126,147,5,788,253,0,2.3,2.5,2.3,2.5,2,0,7,2565,2565],
      [222,41,57,0,99,25,222,0,0,10.7,10.6,10.7,10.6,2,0,4,2385,2385],
      [2067,1056,658,45,303,5,1510,557,0,5,5,5,5,2,0,21,10398,10398],
      [175,65,31,22,38,0,141,15,0,19.6,22,23.6,26.5,1,0,10,3434,4137],
      [230,180,12,41,14,0,247,0,0,1.7,1.6,6,5.6,1,0,3,400,1379],
      [104,32,5,40,28,10,115,0,0,3.3,2.9,4.9,4.5,4,0,2,338,512],
      [177,24,79,6,42,60,211,0,0,16,13.4,16,13.4,2,0,17,2835,2835],
      [62,33,0,12,10,0,22,33,0,9.6,10.8,11.5,13,1,0,2,594,716],
      [487,146,238,53,24,12,45,428,0,2.1,2.2,2.1,2.2,1,0,6,1017,1017],
      [177,40,12,27,38,60,177,0,0,4.5,4.5,5.4,5.4,1,0,8,789,951],
      [105,34,66,10,37,15,162,0,0,7.4,4.8,7.4,4.8,1,0,3,774,774],
      [140,43,68,3,57,0,171,0,0,4.6,3.7,4.6,3.7,4,0,1.5,640,640],
      [230,81,49,37,58,0,225,0,0,2.2,2.2,2.2,2.2,1,0,4,500,500],
      [236,102,84,52,0,0,238,0,0,2.5,2.5,2.8,2.8,1,0,10,601,668],
      [846,322,56,112,234,5,729,0,0,19.1,22.2,19.1,22.2,1,0,13,16179,16179],
      [139,18,21,30,70,0,139,0,0,17.6,17.6,17.6,17.6,2,0,6,2446,2446],
      [51,12,27,0,0,7,7,39,0,6.6,7.3,6.6,7.3,1,0,3,338,338],
      [195,57,32,12,140,0,241,0,0,9.2,7.4,11,8.9,1,0,9,1786,2152],
      [51,32,0,16,7,0,0,55,0,17.6,16.3,21.2,19.7,1,0,3,899,1083],
      [1106,410,267,108,152,25,962,0,0,15,17.2,17.2,19.8,2,0,20,16560,19034],
      [99,13,32,3,42,5,95,0,0,5.3,5.5,6.4,6.7,1,0,4,526,634],
      [56,24,0,9,14,0,47,0,0,7.9,9.4,7.9,9.4,2,0,6,440,440],
      [213,123,91,28,0,0,36,206,0,10.3,9,10.3,9,1,0,7,2185,2185]
    ],
    _tunings =[[
    #         vlow  low   nom   high  vhigh xhigh
    #scale factors:
    'Prec',   6.20, 4.96, 3.72, 2.48, 1.24, _ ],[
    'Flex',   5.07, 4.05, 3.04, 2.03, 1.01, _ ],[
    'Resl',   7.07, 5.65, 4.24, 2.83, 1.41, _ ],[
    'Pmat',   7.80, 6.24, 4.68, 3.12, 1.56, _ ],[
    'Team',   5.48, 4.38, 3.29, 2.19, 1.01, _ ]],
    _doTune = doTune,
    _weighKLOC = weighKLOC,
    _klocWt = klocWt,
    _sdivWeigh = sdivWeigh,
    _split = split,
    _isCocomo = False
    )

def _china(): print(china())