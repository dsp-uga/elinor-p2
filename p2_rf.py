
import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext
from operator import add
import numpy as np
import string

from pyspark.mllib.tree import RandomForest, GradientBoostedTrees
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.feature import ChiSqSelector

sc = pyspark.SparkContext('local[*]',appName="DocClassification")
sqlc = SQLContext(sc)

path = './data/opcodes'

data = sqlc.read.parquet(path)

train_data, test_data = data.randomSplit([.8, .2], seed=42)

train_data = train_data.drop('did') \
            .rdd \
            .map(lambda row: LabeledPoint(row.label, row.features))

test_y = test_data.drop('did') \
            .rdd \
            .map(lambda row: row[1]).zipWithIndex().map(lambda row: (row[1],float(row[0])))
            # .map(lambda row: LabeledPoint(row.label, row.features))
test_data = test_data.drop('did') \
            .rdd \
            .map(lambda row: row[0])

# print("---"*50)
# print("num_train: {}, num_test: {}".format(train_data.count(), test_data.count()))
# print("---"*50)

feat_model = ChiSqSelector(numTopFeatures=50, selectorType='percentile', percentile=0.4, fpr=0.05, fdr=0.05, fwe=0.05).fit(train_data)
train_data = train_data.map(lambda a: LabeledPoint(a.label, feat_model.transform(a.features)))
print(train_data)
# model = RandomForest.trainClassifier(train_data, 10, {}, numTrees=20,featureSubsetStrategy='log2' ,maxDepth=7, maxBins=100)
# model.save(sc, 'rf_model')
# predictions = model.predict(test_data).zipWithIndex().map(lambda row: (row[1], row[0]))
# predictions.foreach(print)
# predict_check = predictions.fullOuterJoin(test_y).map(lambda a: a[1])
# print(predict_check.collect())
#
# metric = MulticlassMetrics(predict_check)
# print(metric.accuracy)