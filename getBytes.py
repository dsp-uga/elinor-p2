import pyspark
from operator import add
from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import NGram, CountVectorizer
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext, Row
import re
from pyspark import StorageLevel
from pyspark.ml.linalg import SparseVector
import numpy
from random import randint

sc = pyspark.SparkContext(appName="malwareClassification")
sqlc = SQLContext(sc)

def saveData(trainData, path):
    trainData.write.parquet(path)

def train(allHex,labels,hashFiles,sc,sqlc,path):

    bytesFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")

    def fun(accum,x):
        return accum+','+x

    bytesFileString = bytesFiles.reduce(fun)
    rdd1= sc.wholeTextFiles(bytesFileString,20)

    bytesRdd = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in allHex.value]).zipWithIndex().map(lambda x: (x[1],x[0]))

    ngramFrame = sqlc.createDataFrame(bytesRdd,["did","1grams"])

    twoGram = NGram(n=2, inputCol="1grams", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    featuresDF = ngramFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['1grams']+x['2grams'])).toDF()

    cv = CountVectorizer(inputCol="docFeatures", outputCol="features",vocabSize=1000)

    featureFitModel = cv.fit(ngramFrame)

    featuresCV = featureFitModel.transform(ngramFrame)

    labelRdd = labels.zipWithIndex().map(lambda x: (x[1],x[0]))

    labelFrame = labelRdd.toDF(["did","label"])

    trainData = ngramFrame.featuresCV(labelFrame,"did")
    trainData.persist(StorageLevel(True, True, False, False, 1))
    saveData(trainData,path)

    trainData.show()
    returm featureFitModel

def test(allHex,hashFiles,sc,sqlc,path,featureFitModel):

    bytesFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/bytes/"+ x+".bytes")
    def fun(accum,x):

        return accum+','+x

    bytesFileString = bytesFiles.reduce(fun)
    rdd1= sc.wholeTextFiles(bytesFileString,20)

    bytesRdd = rdd1.map(lambda x: x[1].split()).map(lambda x: [str(int(word,16)) for word in x if word in allHex.value]).zipWithIndex().map(lambda x: (x[1],x[0]))
    Vec= bytesRdd.map(lambda x: (x[0],createVector(x[1])))
    sparseVec = Vec.map(lambda x: (x[0],SparseVector(256,numpy.nonzero(x[1])[0],x[1][x[1]>0])))

    ngramFrame = sqlc.createDataFrame(sparseVec,["did","1grams"])

    twoGram = NGram(n=2, inputCol="1grams", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    featuresDF = ngramFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['1grams']+x['2grams'])).toDF()

    featuresCV = featureFitModel.transform(ngramFrame)

    testData = featuresCV.drop('docFeatures')
    testData.persist(StorageLevel(True, True, False, False, 1))
    saveData(ngramFrame,path)
    testData.show()


#train data

hashFiles = sc.textFile("gs://uga-dsp/project2/files/X_small_train.txt")
labels = sc.textFile("gs://uga-dsp/project2/files/y_small_train.txt").map(lambda x: int(x))
hashFiles2 = sc.textFile("gs://uga-dsp/project2/files/X_small_test.txt")
allHex= sc.broadcast(set(sc.textFile("gs://dspp1/allHex.txt").collect()))

featureFitModel = train(allHex,labels,hashFiles,sc,sqlc,"gs://dspp1/unigram_train_bytes")

#test Data

test(allHex,hashFiles2,sc,sqlc,"gs://dspp1/unigram_test_bytes",featureFitModel)
