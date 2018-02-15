
import pyspark
from operator import add
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext
from operator import add
import numpy as np
import string

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier


from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext,Row
from operator import add
import numpy as np
import string
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pyspark.sql.functions
from pyspark.ml.linalg import Vectors, VectorUDT


conf = SparkConf().setAppName("MalwareClassification")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '10G'))
sc = SparkContext(conf=conf)
sqlc = SQLContext(sc)

# path = './data/segments/*'
# path = 'gs://elinor-p2/train_features'
# *----------------------------- Set paths -----------------------------------*
train_path = "gs://uga-dsp/project2/files/X_small_train.txt"
test_path = "gs://uga-dsp/project2/files/X_small_test.txt"
train_labels_path = "gs://uga-dsp/project2/files/y_small_train.txt"
test_labels_path = "gs://uga-dsp/project2/files/y_small_test.txt"

# *----------------------------- Get files -----------------------------------*
hashFiles_train = sc.textFile(train_path)
hashFiles_test = sc.textFile(test_path)


# *------------------------ Get labels dataframes ----------------------------*
train_labels = sc.textFile(train_labels_path) \
                    .zipWithIndex() \
                    .map(lambda a: (a[1], a[0]))\
                    .toDF(['did','label'])
test_labels = sc.textFile(test_labels_path) \
                    .zipWithIndex()\
                    .map(lambda a: (a[1], a[0]))\
                    .toDF(['did','label'])


# *---------------------------- Get bytes files ------------------------------*
bytesFiles_train = hashFiles_train.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")
bytesFiles_test = hashFiles_test.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")

# *-------------------- Define accumulator function --------------------------*
def fun(accum,x):
    return accum+','+x

# *------------------ Define train and test bytes files ----------------------*
bytesFileString_train = bytesFiles_train.reduce(fun)
bytesFileString_test = bytesFiles_test.reduce(fun)


# *--------------- Read train/test bytes files to dataframes -----------------*
train_data = sc.wholeTextFiles(bytesFileString_train)\
                .map(lambda x: x[1].split()) \
                .map(lambda x: [word for word in x if len(word)<3]) \
                .zipWithIndex() \
                .map(lambda x: (x[1],x[0])) \
                .toDF(['did', 'words'])
test_data = sc.wholeTextFiles(bytesFileString_test)\
                .map(lambda x: x[1].split()) \
                .map(lambda x: [word for word in x if len(word)<3]) \
                .zipWithIndex() \
                .map(lambda x: (x[1],x[0])) \
                .toDF(['did', 'words'])

train_data.show(10)
test_data.show(10)

# *---------- Define 2Grammer & get train/test documents' 2grams -------------*
grammer = NGram(n=2,inputCol="words",outputCol="grams")
train_data = grammer.transform(train_data)
test_data = grammer.transform(test_data)

train_data.show(10)
test_data.show(10)

# *------------------ Get length of the bytes documents ----------------------*
# train_data = train_data.withColumn("length", length("doc")) \
#                         .drop("doc")
# test_data = test_data.withColumn("length", length("doc")) \
#                         .drop("doc")
#
# train_data.show(10)
# test_data.show(10)
# *--------------------------- Combine 1&2 grams -----------------------------*
train_data = train_data.rdd \
                        .map(lambda x: Row(x['did'],x['words']+x['grams'])) \
                        .toDF(['did','gram_feat'])

test_data = test_data.rdd \
                        .map(lambda x: Row(x['did'],x['words']+x['grams'])) \
                        .toDF(['did','gram_feat'])

train_data.show(10)
test_data.show(10)
# *----------------------- Get Counts of 1&2-grams ---------------------------*
cv = CountVectorizer(inputCol="gram_feat", outputCol = "gram_counts")
train_data = cv.fit(train_data) \
                .transform(train_data) \
                .drop("gram_feat")
test_data = cv.fit(test_data) \
                .transform(test_data) \
                .drop("gram_feat")

train_data.show(10)
test_data.show(10)

# *-------------------- Convert to dense Vector ------------------------------*
train_data = train_data.withColumn("gram_vec", train_data.gram_feat[2]) \
                    .drop("gram_feat")
test_data = test_data.withColumn("gram_vec", test.data.gram_feat[2]) \
                    .drop("gram_feat")

train_data.show(10)
test_data.show(10)

# *--------------------- Combine with length ---------------------------------*
# This part was sourced from this Stack Overflow question:
# https://stackoverflow.com/questions/46556606/adding-a-value-into-a-densevector-in-pyspark
# concat = functions.udf(lambda v, e: Vectors.dense(v + [e]), VectorUDT()) # user defined function to tack doc length to ngram counts
# train_data = train_data.select(concat(train_data.gram_vec, train_data.length).alias('features'))
# test_data = test_data.select(concat(test_data.gram_vec, test_data.length).alias('features'))


# *------------------- Add labels column and drop doc id ---------------------*
train_data = train_data.join(train_labels,['did']) \
                        .drop('did')
test_data = test_data.join(test_labels,['did']) \
                        .drop('did')

# *---------------------- RandomForest Classifier ----------------------------*
rf = RandomForestClassifier(numTrees=20,
                            maxDepth=10,
                            labelCol="label",
                            featuresCol="features",
                            seed=42,
                            predictionCol='prediction',
                            checkPointInterval=10)
model = rf.fit(train_data)
model.save("gs://elinor_temp/models")
test_data = model.transform(test_data)
predictions = test_data.select("prediction").rdd.collect()
with open("gs://elinor_temp/output/predictions.txt", 'w+') as f:
    for i in predictions:
        f.write(i)
# *---------------- Evaluate (Get test/validation accuracy) ------------------*

multi_eval = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                labelCol="label",
                                                metricName='accuracy')
test_accuracy = multi_eval.evaluate(test_data)
print("TEST ACCURACY = {}".format(test_accuracy))
