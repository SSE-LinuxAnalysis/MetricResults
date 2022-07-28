#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import csv
import time

# Required for GPU server (e.g., DGX-1) to select GPU to be used.
os.environ["CUDA_VISIBLE_DEVICES"]="6"

from enum import Enum
class ZEROIMPORTANCE(Enum):
  DISABLED = 1
  ALL = 2
  GPV = 3

##################
### Parameters ###
##################

# Global Paremters
model_str = 'NBC' # GB, NBC, NBG, NN, RF
filter_zero_importance = True
dataset_file = "DS-CVEs.csv" # DS-CVEs.csv, DS-Error.csv
verbosity = 1
# Parameters for Neural Networks
nn_epochs = 400
# Parameters for Random Forest
rf_num_estimators = 500
rf_depth = None
# Parameters for Gradient Boosting
gb_num_estimators = 150
gb_depth = 10

print("Load data")
df_dataset = pd.read_csv(dataset_file, low_memory=False, sep=';')
df_dataset.reset_index(inplace=True, drop=True)
#df_dataset = df_dataset.drop(columns=['Date / Version', 'Source File', 'Element', 'Line No.'])


# Load the subsets
def lookForGlobalVersions(baseline, localVersions):

  globalVersionList = []

  for localname in localVersions:
    found = 0

    # Search for global versions of current 'localname'
    for col in baseline.columns:
      if col.startswith(localname):
        found+=1
        globalVersionList.append(col)

    # If global versions were found drop the local version.
    if found > 1:
      globalVersionList.remove(localname)

  return globalVersionList

print("Prepare metric sub-sets")
# Local-scoped non-variability-aware
LNV = [
       # CyclomaticComplexity
       'McCabe',
       # LoC
       'LoC', 'SCoC',
       # Comments
       'LoC Comment Ratio', 'SCoC Comment Ratio',
       # NestingDepth
       'Classic ND_Avg', 'Classic ND_Max',
       # FanIn
       'Classical Fan-In(global)', 'Classical Fan-In(local)',
       # FanOut
       'Classical Fan-Out(global)', 'Classical Fan-Out(local)',
       # Eigenvector
       'EV Classical Fan-In(global)', 'EV Classical Fan-In(local)',
       'EV Classical Fan-Out(global)', 'EV Classical Fan-Out(local)'
]

# Local-scoped pure-variability-aware

LPV = [
       # VariablesPerFunctionMetric
      'INTERNAL Vars per Function', 'EXTERNAL Vars per Function',
      'EXTERNAL_WITH_BUILD_VARS Vars per Function', 'ALL Vars per Function',
      'ALL_WITH_BUILD_VARS Vars per Function',
      # CyclomaticComplexity
      'CC on VPs',
      # LoC
      # NestingDepth
      'VP ND_Avg', 'VP ND_Max',
      # FanIn
      'VP Fan-In(local)', 'VP Fan-In(global)',
      # FanOut
      'VP Fan-Out(global)', 'VP Fan-Out(local)',
      # Eigenvector
      'EV VP Fan-In(local)', 'EV VP Fan-In(global)',
      'EV VP Fan-Out(global)', 'EV VP Fan-Out(local)',
      # TanglingDegree
      'Full-TD', 'Visible-TD',
      # BlocksPerFunction
      'No. int. blocks x BLOCK_AS_ONE', 'No. int. blocks x SEPARATE_PARTIAL_BLOCKS'
]

# Local-scoped non-variability-aware UNION local-scoped pure-variability-aware
LNV_LPV = LNV + LPV

# Local-scoped combined-variability-aware

LCV = [
       # CyclomaticComplexity
       'McCabe + CC on VPs',
       # LoC
       'SCoF', 'PSCoF', 'LoF', 'PLoF',
       # NestingDepth
       'Combined ND_Max', 'Combined ND_Avg',
       # FanIn
       'DC Fan-In(global)', 'DC Fan-In(local)',
       # FanOut
       'DC Fan-Out(global)', 'DC Fan-Out(local)',
       'DC Fan-Out(global x No Stubs)', 'DC Fan-Out(local x No Stubs)',
       'DC Fan-Out(global x No ext. VPs)', 'DC Fan-Out(local x No ext. VPs)',
       'DC Fan-Out(local x No Stubs x No ext. VPs)', 'DC Fan-Out(global x No ext. VPs)',
       'DC Fan-Out(global x No Stubs x No ext. VPs)',
       # EigenvectorCentrality FanIn
       'EV DC Fan-In(global)', 'EV DC Fan-In(local)',
       # EigenvectorCentrality FanOut
       'EV DC Fan-Out(global)', 'EV DC Fan-Out(local)',
       'EV DC Fan-Out(global x No Stubs)', 'EV DC Fan-Out(local x No Stubs)',
       'EV DC Fan-Out(global x No ext. VPs)', 'EV DC Fan-Out(local x No ext. VPs)',
       'EV DC Fan-Out(global x No Stubs x No ext. VPs)', 'EV DC Fan-Out(local x No Stubs x No ext. VPs)',
       # UndisciplinedPreprocessorUsage
       'Undisciplined CPP'
]

# Local-scoped combined-variability-aware
# UNION local-scoped pure-variability-aware
# UNION local-scoped non-variability-aware
LCV_LPV_LNV = LCV + LPV + LNV

# Global-scoped pure-variabilty-aware
# Replaces all local variants for globals
# If there is no global variant, it includes the local
GPV = lookForGlobalVersions(df_dataset, LPV)

# Global-scoped combined-variabilty-aware
# Replaces all local variants for globals
# If there is no global variant, it includes the local
GCV = lookForGlobalVersions(df_dataset, LCV)

# Global-scoped combined-variability-aware
# UNION Global-scoped pure-variability-aware
# UNION local-scoped non-variability-aware
# list-set-cast to avoid redundant features in list
GCV_GPV_LNV = list(set(GCV + GPV + LNV))

# All metrics included
# list-set-cast to avoid redundant features in list
ALL = list(set(LNV + LPV + LCV + GPV + GCV))

# OneHotEncoded CATEGORICAL FEATURES
CATEGORICAL = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16' ]

"""## Load the subsets"""

def getRFESubsets(path):
  df = pd.read_csv(path, index_col=None)
  patterns = ["LNV", "LPV", "LCV", "GPV", "GCV", "ALL"]

  return [list(df[df["StartSubset"] == pattern]["Feature"]) for pattern in patterns]

LNV_RFE, LPV_RFE, LCV_RFE, GPV_RFE, GCV_RFE, ALL_RFE = getRFESubsets('RFE_reduced_subsets_final.csv')

# Remove duplicated columns
LCV = list(dict.fromkeys(LCV))
GCV = list(dict.fromkeys(GCV))

# Dictionary that contains the 12 different subsets
datasets = {
    "LNV": LNV,
    "LPV": LPV,
    "LCV": LCV,
    "GPV": GPV,
    "GCV": GCV,
    "ALL": ALL,
    "LNV_RFE": LNV_RFE,
    "LPV_RFE": LPV_RFE,
    "LCV_RFE": LCV_RFE,
    "GPV_RFE": GPV_RFE,
    "GCV_RFE": GCV_RFE,
    "ALL_RFE": ALL_RFE
}

samplings = [
             "Over",
             # "Under",
             "Smote",
             "Off"
]

# Load zero important features
ZERO_IMPORTANCE_ALL = pd.read_csv('ZERO_IMPORTANCE_ALL.csv', index_col=0).values.flatten()
ZERO_IMPORTANCE_GPV = pd.read_csv('ZERO_IMPORTANCE_GPV.csv', index_col=0).values.flatten()

# remove the outer Validation from data set
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# split data set in stratified manner
#data, outerValidation = train_test_split(df_dataset, test_size=0.1, random_state=42, shuffle=True, stratify=df_dataset['Error'])
# New data set (CVEs), which wasn't used for optimization and model built -> do not remove anything
data = df_dataset

"""# Auxiliary"""

def learnNormalization(train, order=2, axis=0):
  l2 = np.atleast_1d(np.linalg.norm(train, order, axis))
  l2[l2 == 0] = 1
  return l2

def normalize(x, norm, axis=0):
  #return x / norm
  return x / np.expand_dims(norm, axis)

def undersample(x, targetSize, factor=1):
  x = resample(x, replace=False, n_samples=targetSize*factor, random_state=42)
  return x

def oversample(x, targetSize, factor=1):
  x = resample(x, replace=True, n_samples=targetSize*factor, random_state=42)
  return x

def smote(x, y):
  sm = SMOTE(sampling_strategy=1.0, k_neighbors=5, n_jobs=-1)
  x, y = sm.fit_resample(x,y)
  return x, y

def preparationPipeline(train, test, currentSubset, sampling='Off', samplingFactor=1, categorical=False, samplingArgs={}, zero_importance=ZEROIMPORTANCE.DISABLED):

  # BUILD TRAINING DATA

  # Get the current subset
  train_x = train[currentSubset]
  trainColumns = train_x.columns.values

  # Learn and apply the normalization
  l2 = learnNormalization(train_x)
  train_x = normalize(train_x, norm=l2)

  # Add categorical features
  if categorical:
    train_x = pd.concat([train[CATEGORICAL], train_x], axis=1)

  # Get y
  train_y = train['Error']
  labelsColumns = 'Error'

  # Resampling
  if sampling != 'Off':
    if sampling == 'Smote':
      train_x, train_y = smote(train_x, train_y)
    else:

      train_x_y = pd.concat([train_x, train_y], axis=1)

      noDefect = train_x_y[train_x_y['Error'] == 0]
      defect = train_x_y[train_x_y['Error'] == 1]

      if sampling == 'Under':
        noDefect = undersample(noDefect, targetSize=defect.shape[0], factor=samplingFactor)
      elif sampling == 'Over':
        defect = oversample(defect, targetSize=noDefect.shape[0], factor=samplingFactor)

      train_x_y = pd.concat([noDefect, defect], axis=0)
      train_x = train_x_y.drop(labels='Error', axis=1)
      train_y = train_x_y['Error']

  # BUILD TEST DATA
  test_x = test[currentSubset]
  test_x = normalize(test_x, norm=l2)

  # Drop features with Zero Importance (figured out in previous experiments)
  if zero_importance==ZEROIMPORTANCE.ALL:
    train_x = train_x.drop(labels=ZERO_IMPORTANCE_ALL, axis=1)
    test_x = test_x.drop(labels=ZERO_IMPORTANCE_ALL, axis=1)
  elif zero_importance==ZEROIMPORTANCE.GPV:
    train_x = train_x.drop(labels=ZERO_IMPORTANCE_GPV, axis=1)
    test_x = test_x.drop(labels=ZERO_IMPORTANCE_GPV, axis=1)

  if categorical:
    test_x = pd.concat([test[CATEGORICAL], test_x], axis=1)

  test_y = test['Error']

  # return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy(), trainColumns, labelsColumns
  return train_x, train_y, test_x, test_y, trainColumns, labelsColumns

k_fold_outer_seeds = [42, 512, 51234, 641, 67323, 2, 856, 12, 291, 4865]

"""# Evaluate Classifiers"""

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from NeuralNetwork import *
from sklearn.metrics import *
from datetime import datetime

start_timestamp_overall = datetime.now()
k_fold_iterations = 1
results = pd.DataFrame(columns=['Classifier', 'Cluster', 'Sampling', 'Zero Importance', 'K-fold Outer Iteration', 'K-fold Inner Iteration', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Duration'])

models = {
  'GB': GradientBoostingClassifier,
  'NBC': ComplementNB,
  'NBG': GaussianNB,
  'NN': NeuralNet,
  'RF': RandomForestClassifier
}


for subset in datasets:
  # Start timer for current subset evaluation
  print("Starts analysis of: " + subset)
  start_timestamp_subset = datetime.now()

  for sampling in samplings:

    # Start timer for current sampling method evaluation
    start_timestamp_sampling = datetime.now()

    for k_fold_outer_counter in range(k_fold_iterations):

      # Start timer for current k-fold evaluation
      start_timestamp_k_fold = datetime.now()

      randomState = 42 if k_fold_outer_counter >= 10 else k_fold_outer_seeds[k_fold_outer_counter]
      kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=randomState)

      k_fold_inner_counter = 0
      for train_index, test_index in kf.split(data, data['Error']):
        if subset == "GPV" and filter_zero_importance:
          z_importance = ZEROIMPORTANCE.GPV
        elif subset == "ALL" and filter_zero_importance:
          z_importance = ZEROIMPORTANCE.ALL
        else:
          z_importance = ZEROIMPORTANCE.DISABLED

        # Build the train/test data for current k-fold iteration
        train = data.iloc[train_index] #.apply(pd.to_numeric)
        test = data.iloc[test_index]   #.apply(pd.to_numeric)
        train_x, train_y, test_x, test_y, trainColumns, labelsColumns = preparationPipeline(train, test, datasets[subset], sampling=sampling, samplingFactor=1, categorical=False, zero_importance = z_importance)

        # Create the model by calling current set estimator
        if model_str == 'GB':
          model = models[model_str](n_estimators=gb_num_estimators, max_depth=gb_depth, n_iter_no_change=10, random_state=42, verbose=verbosity)
        elif model_str == 'NBG' or model_str == 'NBC':
          model = models[model_str]()
        elif model_str == 'NN':
          model = models[model_str]()
          opt = Optimizer(model, mb=128, lr=0.0001)
        elif model_str == 'RF':
          model = models[model_str](n_estimators=rf_num_estimators, max_depth=rf_depth, n_jobs=-1, random_state=42, verbose=verbosity) # WITHOUT random state=42

        start_model_timestamp = datetime.now()

        # Fit to train data and predict on test set
        if model_str != 'NN':
          y_pred = model.fit(train_x, train_y).predict(test_x)

          # Create the final metrics
          accuracy = accuracy_score(test_y, y_pred)
          precision = precision_score(test_y, y_pred)
          recall = recall_score(test_y, y_pred)
          f1 = f1_score(test_y, y_pred)
          auc = roc_auc_score(test_y, model.predict_proba(test_x)[:,1], average='micro')
        else:
          accuracy, precision, recall, f1, auc = opt.run(train_x, train_y, test_x, test_y, nn_epochs, verbose=verbosity)

        duration = (datetime.now() - start_model_timestamp).seconds

        results = results.append([{'Classifier': model_str, 'Cluster': subset, 'Sampling': sampling, 'Zero Importance': z_importance.name,
                                   'K-fold Outer Iteration': k_fold_outer_counter, 'K-fold Inner Iteration': k_fold_inner_counter,
                                   'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': auc,
                                   'Duration': duration}])

        k_fold_inner_counter +=1

      if k_fold_iterations > 1:
        print("Duration of " + str(k_fold_outer_counter) + " (outer) k-fold evaluation was: " + str(datetime.now() - start_timestamp_k_fold))

    print("Duration of current sampling (" + str(sampling) + ") evaluation was: " + str(datetime.now() - start_timestamp_sampling))

  print("Duration of current subset (" + subset + ") evaluation was: " + str(datetime.now() - start_timestamp_subset))
  print("#################################")
  timestamp = datetime.now()
  results.to_csv("results/" + model_str + "/results_" + model_str + "_" + subset + "_"+ str(timestamp) + ".csv", sep=';', decimal=',')
  results = pd.DataFrame(columns=['Classifier', 'Cluster', 'Sampling', 'Zero Importance', 'K-fold Outer Iteration', 'K-fold Inner Iteration', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Duration'])

print("#################################")
print("Duration of overall evaluation was: " + str(datetime.now() - start_timestamp_overall))
