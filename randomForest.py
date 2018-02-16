import pyspark
from operator import add
from pyspark import SparkConf,SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit

sc = pyspark.SparkContext(appName="malwareClassification")
sqlc = SQLContext(sc)

# # --- Loading and Train Data---

dataForModel =sqlc.read.parquet("gs://dspp1/final_small_train_features")

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dataForModel)

print("--------------------labelIndexer Done-----------------------")

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

print("--------------------RandomForest Model Created-----------------------")

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=labelIndexer.labels)

print("--------------------LabelConverter Done-----------------------")

pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

print("--------------------Pipeline Created-----------------------")

model = pipeline.fit(dataForModel)

print("--------------------Training Completed-----------------------")

#----Testing Starts----

testingData =sqlc.read.parquet("gs://dspp1/final_small_test_features")

print("--------------------Testing Data Loaded-----------------------")

predictions = model.transform(testingData)


print("--------------------Model Transformed-----------------------")

predicted = predictions.select("predictedLabel")
predicted.write.parquet('gs://dspp1/small_predictions")
