from __future__ import division,print_function
import  sys
sys.dont_write_bytecode = True
from lib import *
import numpy as np
import pandas
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy.stats.stats import skew

__author__ = 'panzer'

def make_formula(model, col_names):
  labels = []
  for index, col_name in enumerate(model.indep):
    if col_name not in col_names: continue
    if model.is_continuous[index]:
      labels.append(col_name)
    else:
      labels.append("C(%s, levels=[0,1,2,3,4,5,6])"%col_name)
  formula = "%s ~ %s"%(model.less[0], " + ".join(labels))
  return formula


def lin_reg_pruned(model, test_row, rows):
  headers = model.indep + model.less
  train_data = [row.cells[:] for row in rows]
  continuous_variables = [decision for decision in model.decisions if model.is_continuous[decision]]
  transforms = get_transform_funcs(train_data, continuous_variables)
  for row in train_data:
    transform_row(row, continuous_variables, transforms)
  columnar_data = make_columnar(train_data, model.decisions)
  column_size = len(columnar_data)
  row_size = len(rows)
  if row_size < column_size * 10:
    pruned_column_size = int(row_size/10)
    efforts = [effort(model, row) for row in rows]
    ranks = []
    for index, column in enumerate(columnar_data):
      is_continuous = model.is_continuous[index]
      ranks.append((covariance(column, efforts, is_continuous), model.indep[index]))
    pruned_headers = [rank[1] for rank in sorted(ranks)[:pruned_column_size]]
  else:
    pruned_headers = headers
  df = pandas.DataFrame(train_data, columns=headers)
  lin_model = smf.ols(formula=make_formula(model, pruned_headers), data=df).fit()
  test_data = transform_row(test_row.cells[:], continuous_variables, transforms)
  df_test = pandas.DataFrame([test_data], columns=headers)
  return lin_model.predict(df_test)[0]

def get_transform_funcs(train, cols):
  transform_funcs = []
  for col in cols:
    vector = [row[col] for row in train]
    transforms = [
      (skew(vector, bias=False), "none"),
      (skew(log_transform(vector), bias=False), "log"),
      (skew(sqrt_transform(vector), bias=False), "sqrt")
    ]
    best_transform = sorted(transforms)[0][1]
    transform_funcs.append(best_transform)
  return transform_funcs

def transform_row(row, cols, transforms):
  for col, transform in zip(cols, transforms):
    if transform == "log":
      row[col] = math.log(row[col])
    elif transform == "sqrt":
      row[col] = math.sqrt(row[col])
    elif transform == "none":
      continue
    else:
      raise RuntimeError("Unknown transformation type : %s"%transform)
  return row

def log_transform(vector):
  transforms = []
  for one in vector:
    if one == 0:
      transforms.append(-float("inf"))
    else:
      transforms.append(math.log(one))
  return transforms

def sqrt_transform(vector):
  return [math.sqrt(one) for one in vector]

def make_columnar(rows, columns):
  column_data = []
  for column in columns:
    column_data.append([row[column] for row in rows])
  return column_data

def get_column(rows, column_index):
  return [row[column_index] for row in rows]

def covariance(x_vector, y_vector, is_continuous):
  if is_continuous:
    return abs(spearmanr(x_vector, y_vector, nan_policy="omit")[0])
  else:
    x_vector_str = map(str, x_vector)
    labels = list(set(x_vector_str))
    y_vector_str = list(pandas.cut(y_vector, len(labels), labels=labels).get_values())
    return abs(spearmanr(x_vector_str, y_vector_str, nan_policy="omit")[0])
