
# Malware Classification
## Team: Elinor
#### Members: 
* Rajeswari Sivakumar
* Nick Klepp
* Vyom Shrivastava

Classifier : Naive - Bayes

## Technologies Used:
* Apache Spark
  ** Packages: spark.ml, spark.sql
* pySpark
* Python 3.5

## Overview:
This repository contains implementaion of various `spark.ml` based pre-processing models and classifiers built using Apache Spark to classify files into malware groups. This is done as a project for the course CSCI 8360: Data Science Practicum.

## Dataset:
The datasets we used to train the classifiers are provided by Dr. Shannon Quinn for the course CSCI 8360: Data Science Practicum. We used 2 datasets to to train and tests our model which were indexed as lists of hashes that corresponded to names fo `.asm` and `.bytes` files. Each documents belonged to one of 9 possibe malware classes
1. Small dataset to develop and test models. 
2. Large dataset to train final model and generate predictions for test set.

## Project Summary:
Given several `.bytes` and `.asm` files for several  the current project aims to classify documents from 1 of 9 groups. Each document can only have one class. To this end, we conducted research and derived inspiration from Kaggle competition winners that this project is based on. In their presentation, they note several features to be advantages to them:
  - Unigram through 4grams for `bytes` files.
  - Unigram through 4grams for `opcodes` in `asm` files. (`opcodes` are translations of bytes files that correspond to specific operation
  - Denoted segments in `asm` files such as the `HEADER`

We trained our features on the `pyspark` `ml.classifaction.RandomForestClassifier` starting the default settings of the classifier, and then tweaking to try upto 30 trees with max depth of 8. We also trained a `ml.classification.NaiveBayesClassifier` but were not successful in improving the results with this approach either

Due to memory issues we were not able to use all the features initially. Thus we limited ourselves to subsets of these features and tried various combinations of subsets. 

Inspite of this work-around and also tweaking the classifiers in multiple ways we were not able to exceed 27.6% accuracy on the final test set.

##Executions:

1. We generate features based on the code found in the `getFeatures.py` file. They can be saved as parquet files to be read in later.
2. We used the generated features to train naive bayes and random forest classifiers. Examples can be seen in `naive3.py`
