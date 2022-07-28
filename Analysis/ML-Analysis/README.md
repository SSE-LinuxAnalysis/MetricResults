# ML-Scripts-Dissertation

The data analysis scripts of my dissertation

## Files

* `ML-Analysis.py`: Main analysis script. This script can be used to start the various analyses.
* `NeuralNetwork.py`: Additional script that implements the Neural Network. Will be executed as part of `ML-Analysis.py`, if the neural network analysis was selected.
* `testGPU.py`: May be used to check whether GPUs are configured correctly and can be used by Tensorflow.
* `RFE_reduced_subsets_final.csv`: The list of metrics retained by each RFE-cluster.
* `ZERO_IMPORTANCE_ALL.csv`: The list of unimportant metrics of the **ALL** cluster.
* `ZERO_IMPORTANCE_GPV.csv`: The list of unimportant metrics of the **GPV** cluster.
Please note that all `CSV` files are encoded in German, i.e., `;` was used as separator.

## Parameters

`ML-Analysis.py` provides the following parameters:

* `os.environ["CUDA_VISIBLE_DEVICES"]` (Line 11): Allows to select which GPU shall be used by Tensorflow (0-7 for DGX-1). This is only relevant for the Neuronal Network analysis. See next parameter for more details.
* `model_str` (Line 24): Defines which analysis shall be conducted:
  * `GB`: [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  * `NBC`: [Complement Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
  * `NBG`: [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
  * `NN`: Neural Network using [Keras by Tensorflow](https://keras.io/api/models/model/)
  * `RF`: [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* `filter_zero_importance` (Line 25):
  * `True`: Uses `ZERO_IMPORTANCE_*.csv` to exclude irrelevant metrics from analysis to speed up analysis and to improve accuracy. Only relevant for ALL and GPV analyses as they make use of huge metric clusters. Also not considered during RFE-based analysis.
  * `False`: Don't use this optimization
* `dataset_file` (Line 26): Specifies which data set shall be analyzed.
* `verbosity` (Line 27): Specifies how much log data shall be printed (0-3).
* `datasets` (Lines 182 - 195): Specifies which metric clusters should be analyzed.
* `samplings` (Lines 197 - 202): Specifies which sampling strategies shall be used.

### Parameters Specific for Gradient Boosting

* `gb_num_estimators` (Line 34): Specifies how many estimators shall be used.
* `gb_depth` (Line 35): Specifies the maximal depth of each estimator.

### Parameters Specific for Neural Networks

* `nn_epochs` (Line 29): Specifies how many epochs shall be trained.

### Parameters Specific for Random Forest

* `rf_num_estimators` (Line 31): Specifies how many estimators shall be used.
* `rf_depth` (Line 32): Specifies the maximal depth of each estimator.
