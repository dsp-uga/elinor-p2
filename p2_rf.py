
import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext
from operator import add
import numpy as np
import string

# from pyspark.mllib.tree import RandomForest, GradientBoostedTrees
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
# from pyspark.mllib.feature import ChiSqSelector

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier


rom pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf, withColumn
from pyspark.sql import SQLContext,Row
from operator import add
import numpy as np
import string
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
            .zipWithIndex()\
            .map(lambda a: (a[1], a[0]))\
            .toDF(['did', 'doc'])
test_data = sc.wholeTextFiles(bytesFileString_test)\
            .zipWithIndex()\
            .map(lambda a: (a[1], a[0]))\
            .toDF(['did', 'doc'])

train_data.show(10)
test_data.show(10)

# *---------- Define Tokenizer and tokenize train/test documents -------------*

toker = Tokenizer(inputCol = "doc",outputCol = "words")
train_data = toker.transform(train_data)
test_data = toker.transform(test_data)

# *---------- Define 2Grammer & get train/test documents' 2grams -------------*
grammer = NGram(n=2,inputCol="words",outputCol="grams")
train_data = grammer.transform(train_data)
train_data = grammer.transform(train_data)

# *------------------ Get length of the bytes documents ----------------------*
train_data train_data.withColumn("length", length("doc"))
test_data test_data.withColumn("length", length("doc"))

# *--------------------

# *------------------ Get Counts of 1&2-grams ----------------------*

countVec = CountVectorizer()





rf = RandomForestClassifier(numTrees=20, maxDepth=10, labelCol="label",featuresCol="features", seed=42, predictionCol='predictions')
model = rf.fit(train_data)
predictions = model.transform(test_data)
