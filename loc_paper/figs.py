from __future__ import print_function, division
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

__author__ = "panzer"

def loc_error():
  data = OrderedDict([
    ("COCOMO2", {
      "NASA10": (43, 0),
      "COC05": (46, 0),
      "NASA93": (39, 0),
      "COC81": (32, 0)
    }), ("0.2:COCOMO2", {
      "NASA10": (38, 1),
      "COC05": (46, 2),
      "NASA93": (39, 1),
      "COC81": (32, 1)
    }), ("0.4:COCOMO2", {
      "NASA10": (45, 2),
      "COC05": (48, 5),
      "NASA93": (40, 2),
      "COC81": (36, 2)
    }), ("0.6:COCOMO2", {
      "NASA10": (49, 4),
      "COC05": (50, 8),
      "NASA93": (42, 3),
      "COC81": (43, 4)
    }), ("0.8:COCOMO2", {
      "NASA10": (49, 6),
      "COC05": (63, 12),
      "NASA93": (44, 6),
      "COC81": (49, 6)
    }), ("1.0:COCOMO2", {
      "NASA10": (50, 8),
      "COC05": (65, 18),
      "NASA93": (51, 8),
      "COC81": (50, 9)
    })
  ])
  methods = data.keys()
  padding = 0.15
  width = (1-padding) / len(methods)
  fig, ax = plt.subplots()
  datasets = ["NASA10", "COC05", "NASA93", "COC81"]
  colors = ['b', 'm', 'g', 'c', 'r', 'y']
  blocks = []
  for i, method in enumerate(methods):
    ind = np.arange(len(datasets))
    means = []
    variances = []
    for dataset in datasets:
      means.append(data[method][dataset][0])
      variances.append(data[method][dataset][1])
    rects = ax.bar((ind + i*width) + padding/2, means, width, color=colors[i], yerr=variances, ecolor='k')
    blocks.append(rects[0])
  ax.set_ylabel('Mean Relative Error %')
  ax.set_xlabel('Datasets')
  ax.set_title('MRE for effort while varying LOC in different datasets')
  ax.set_xticks(np.arange(len(datasets)) + len(methods)*width/2 + padding/2)
  ax.set_xticklabels(datasets)
  ax.legend(blocks, methods, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=7, fancybox=True, shadow=True)
  plt.savefig("loc_paper/mre.png", bbox_inches='tight',dpi=100)
  plt.clf()

loc_error()
