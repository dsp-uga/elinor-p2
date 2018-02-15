import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext,Row
from operator import add
import numpy as np
import string
from pyspark.ml.feature import HashingTF, IDF, Tokenizer,NGram,CountVectorizer
from pyspark.ml.feature import StringIndexer
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import VectorUDT as VectorUDTML
from pyspark.sql.functions import udf

as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDTML())

DEBUG = False

sc = pyspark.SparkContext('local[*]',appName="DocClassification")
sqlc = SQLContext(sc)

if DEBUG :
    DATA_PATH  = "data/p2/X_small_train_testing.txt" 
    LABEL_PATH = "data/p2/y_small_train_testing.txt" 

    TEST_DATA  = "data/p2/X_small_test_testing.txt" 
    TEST_LABS  = "data/p2/y_small_test_testing.txt"
else:
    DATA_PATH  = "gs://uga-dsp/project2/files/X_small_train.txt" 
    LABEL_PATH = "gs://uga-dsp/project2/files/y_small_train.txt" 

    TEST_DATA  = "gs://uga-dsp/project2/files/X_small_test.txt" 
    TEST_LABS  = "gs://uga-dsp/project2/files/y_small_test.txt"

#------------------------------DEAL WITH TRAIN DATA-----------------------------#

if DEBUG :
    wtf = sc.textFile(TEST_DATA)\
            .map(lambda x: "data/p2/bytes/" + x + ".bytes")\
            .reduce(lambda accum,x: accum + "," + x)

else :
    wtf = sc.textFile(TEST_DATA)\
            .map(lambda x: "gs://uga-dsp/project2/data/bytes/" + x + ".bytes")\
            .reduce(lambda accum,x: accum + "," + x)
    
data = sc.wholeTextFiles(wtf,20)\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0][1]))\
              .toDF(['did','doc'])

test_labs = sc.textFile(LABEL_PATH)\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0]))\
              .toDF(['did','label'])


toker = Tokenizer(inputCol = "doc",outputCol = "words")
data = toker.transform(data)
grammer = NGram(n=2,inputCol="words",outputCol="grams")
data = grammer.transform(data).drop('doc')

data = data.rdd.map(lambda x: Row(x['did'],x['words']+x['grams'])).toDF(['did','features'])

cv = CountVectorizer(inputCol="features", outputCol="featureVecs")
data = cv.fit(data).transform(data)

data = data.join(labels,['did'])

indexer = StringIndexer(inputCol="lab", outputCol="indexedLabel")
data = indexer.fit(data).transform(data)

#------------------------------DEAL WITH TEST DATA-----------------------------#

if DEBUG :
    wtf = sc.textFile(TEST_DATA)\
            .map(lambda x: "data/bytes/" + x + ".bytes")\
            .reduce(lambda accum,x: accum + "," + x)

else:
    wtf = sc.textFile(TEST_DATA)\
            .map(lambda x: "gs://uga-dsp/project2/data/bytes" + x + ".bytes")\
            .reduce(lambda accum,x: accum + "," + x)
    
test_data = sc.wholeTextFiles(wtf)\
                  .zipWithIndex()\
                  .map(lambda x: (x[1],x[0][1]))\
                  .toDF(['did','doc'])
    
test_labs = sc.textFile(LABEL_PATH)\
         .zipWithIndex()\
         .map(lambda x: (x[1],x[0]))\
         .toDF(['did','label'])


toker = Tokenizer(inputCol = "doc",outputCol = "words")
test_data = toker.transform(test_data)
grammer = NGram(n=2,inputCol="words",outputCol="grams")
test_data = grammer.transform(test_data).drop('doc')

test_data = test_data.rdd.map(lambda x: Row(x['did'],x['words']+x['grams'])).toDF(['did','features'])

cv = CountVectorizer(inputCol="features", outputCol="featureVecs")
test_data = cv.fit(test_data).transform(test_data)

test_data = test_data.join(test_labels,['did'])

indexer = StringIndexer(inputCol="lab", outputCol="indexedLabel")
test_data = indexer.fit(test_data).transform(test_data)

#------------------------------FIT RANDOM FOREST-----------------------------#

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="featureVecs", numTrees=10)
model = rf.fit(data)


#------------------------------FIT NAIVE BAYES FOREST-----------------------------#

data2 = data.rdd\
            .map(lambda x: tuple(x))\
            .map(lambda x: (x[3],x[2]))\
            .map(lambda x: LabeledPoint(x[0],MLLibVectors.fromML(x[1])))\
            .toDF()

test_data2 = test_data.rdd\
                      .map(lambda x: tuple(x))\
                      .map(lambda x: (x[3],x[2]))\
                      .map(lambda x: LabeledPoint(x[0],MLLibVectors.fromML(x[1])))\
                      .toDF()

data2 = data2.withColumn("features", as_ml("features"))
test_data2 = test_data2.withColumn("features", as_ml("features"))

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(data2)

predictionAndLabel = test_data2.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))
