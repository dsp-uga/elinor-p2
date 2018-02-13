
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
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
# from pyspark.mllib.feature import ChiSqSelector

from pyspark.ml.feature import ChiSqSelector


sc = pyspark.SparkContext('local[*]',appName="DocClassification")
sqlc = SQLContext(sc)

path = './data/opcodes'

data = sqlc.read.parquet(path).drop('did')
data = MLUtils.convertVectorColumnsToML(data)
data = data.withColumn("label", data["label"].cast("double"))
data.show()

feat_model = ChiSqSelector(percentile=0.5, outputCol="selectedFeatures")
feat_model = feat_model.fit(data).save("./models/chisq")
data = feat_model.transform(data).drop('features')

data.show()

train_data, test_data = data.randomSplit([.8, .2], seed=42)

train_data = train_data.drop('did')
test_y = test_data.drop('did').drop('features')
test_data = test_data.drop('did')

model = RandomForest.trainClassifier(train_data, 10, {}, numTrees=20,featureSubsetStrategy='log2' ,maxDepth=7, maxBins=100)
model.save(sc, 'rf_model')
predictions = model.predict(test_data).zipWithIndex().map(lambda row: (row[1], row[0]))
predictions.foreach(print)
predict_check = predictions.fullOuterJoin(test_y).map(lambda a: a[1])
print(predict_check.collect())

metric = MulticlassMetrics(predict_check)
print(metric.accuracy)
