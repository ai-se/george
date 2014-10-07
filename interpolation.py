"""
Source file to perform interpolation on the nasa93 data
"""

import sys
from lib import *
from nasa93 import *
from settings import *
from where2 import *

def interpolate(m):
  launchWhere2(m)

@go
def _interpolate():
  interpolate(nasa93)
