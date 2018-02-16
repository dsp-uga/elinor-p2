import pyspark
from operator import add
from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import NGram, CountVectorizer
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext, Row
import re
from pyspark import StorageLevel

sc = pyspark.SparkContext(appName="malwareClassification")
sqlc = SQLContext(sc)

def saveData(trainData, path):
    trainData.write.parquet(path)

def train(labels,hashFiles,sc,sqlc,path):

    bytesFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")

    def fun(accum,x):
        return accum+','+x

    bytesFileString = bytesFiles.reduce(fun)
    rdd1= sc.wholeTextFiles(bytesFileString,20)

    bytesRdd = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if len(word)<3]).zipWithIndex().map(lambda x: (x[1],x[0]))

    ngramFrame = sqlc.createDataFrame(bytesRdd,["docId","1grams"])

    twoGram = NGram(n=2, inputCol="1grams", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    featuresDF = ngramFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['1grams']+x['2grams'])).toDF()

    cv = CountVectorizer(inputCol="docFeatures", outputCol="features",vocabSize=2000)

    featureFitModel = cv.fit(featuresDF)

    featuresCV = featureFitModel.transform(featuresDF)

    labelRdd = labels.zipWithIndex().map(lambda x: (x[1],x[0]))

    labelFrame = labelRdd.toDF(["did","label"])

    trainData = featuresCV.join(labelFrame,"did").drop('docFeatures')
    trainData.persist(StorageLevel(True, True, False, False, 1))
    saveData(trainData,path)

    trainData.show()
    return featureFitModel


def test(hashFiles,sc,sqlc,path,featureFitModel):

    bytesFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")

    def fun(accum,x):
        return accum+','+x

    bytesFileString = bytesFiles.reduce(fun)

    rdd1= sc.wholeTextFiles(bytesFileString,20)

    bytesRdd = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if len(word)<3]).zipWithIndex().map(lambda x: (x[1],x[0]))

    ngramFrame = sqlc.createDataFrame(bytesRdd,["docId","1grams"])

    twoGram = NGram(n=2, inputCol="1grams", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    featuresDF = ngramFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['1grams']+x['2grams'])).toDF()

    featuresCV = featureFitModel.transform(featuresDF)

    testData = featuresCV.drop('docFeatures')
    testData.persist(StorageLevel(True, True, False, False, 1))
    saveData(testData,path)
    testData.show()


#train data

hashFiles = sc.textFile("gs://uga-dsp/project2/files/X_small_train.txt")
labels = sc.textFile("gs://uga-dsp/project2/files/y_small_train.txt")
hashFiles2 = sc.textFile("gs://uga-dsp/project2/files/X_small_test.txt")

featureFitModel = train(labels,hashFiles,sc,sqlc,"gs://dspp1/small_train_bytes")

#test Data

test(hashFiles2,sc,sqlc,"gs://dspp1/small_test_bytes",featureFitModel)
