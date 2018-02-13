
import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext
from operator import add
import numpy as np
import string

from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics


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

model = RandomForest.trainClassifier(train_data, 10, {}, 10)

predictions = model.predict(test_data).zipWithIndex().map(lambda row: (row[1], row[0]))

predict_check = predictions.fullOuterJoin(test_y).map(lambda a: a[1])
print(predict_check.collect())

metric = MulticlassMetrics(predict_check)
print(metric.accuracy)
