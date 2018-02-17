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

def train(opcodes,labels,hashFiles,sc,sqlc,path):

    asmFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/asm/"+ x+".asm")

    def fun(accum,x):
        return accum+','+x

    asmFileString = asmFiles.reduce(fun)
    rdd1= sc.wholeTextFiles(asmFileString,20)

    opcodesInDoc = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in opcodes.value]).zipWithIndex().map(lambda x: (x[1],x[0]))

    ngramFrame = sqlc.createDataFrame(opcodesInDoc,["docId","opcodes"])

    twoGram = NGram(n=2, inputCol="opcodes", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    threeGram = NGram(n=3, inputCol="opcodes", outputCol="3grams")
    ngramFrame= threeGram.transform(ngramFrame)

    fourGram = NGram(n=4, inputCol="opcodes", outputCol="4grams")
    ngramFrame = fourGram.transform(ngramFrame)

    def getSegment(x):
        templist=[]
        for line in x:
            l= re.findall(r'\w+:?(?=:)',line)
            if l:
                templist.append(l[0])
        return templist

    segments = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].splitlines())).map(lambda x: (x[0],getSegment(x[1]))).toDF(["docId","segments"])

    featureFrame= ngramFrame.join(segments, "docId")

    featuresDF = featureFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['opcodes']+x['2grams']+x['3grams']+x['4grams']+x['segments'])).toDF()

    cv = CountVectorizer(inputCol="docFeatures", outputCol="features",vocabSize=5000)

    featureFitModel = cv.fit(featuresDF)

    featuresCV = featureFitModel.transform(featuresDF)

    labelRdd = labels.zipWithIndex().map(lambda x: (x[1],x[0]))

    labelFrame = labelRdd.toDF(["did","label"])

    trainData = featuresCV.join(labelFrame,"did").drop('docFeatures')
    trainData.persist(StorageLevel(True, True, False, False, 1))
    saveData(trainData,path)

    trainData.show()
    return featureFitModel


def test(opcodes,hashFiles,sc,sqlc,path,featureFitModel):

    asmFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/asm/"+ x+".asm")

    def fun(accum,x):
        return accum+','+x

    asmFileString = asmFiles.reduce(fun)

    rdd1= sc.wholeTextFiles(asmFileString,20)

    opcodesInDoc = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in opcodes.value]).zipWithIndex().map(lambda x: (x[1],x[0]))

    ngramFrame = sqlc.createDataFrame(opcodesInDoc,["docId","opcodes"])

    twoGram = NGram(n=2, inputCol="opcodes", outputCol="2grams")
    ngramFrame = twoGram.transform(ngramFrame)

    threeGram = NGram(n=3, inputCol="opcodes", outputCol="3grams")
    ngramFrame= threeGram.transform(ngramFrame)

    fourGram = NGram(n=4, inputCol="opcodes", outputCol="4grams")
    ngramFrame = fourGram.transform(ngramFrame)

    def getSegment(x):
        templist=[]
        for line in x:
            l= re.findall(r'\w+:?(?=:)',line)
            if l:
                templist.append(l[0])
        return templist

    segments = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].splitlines())).map(lambda x: (x[0],getSegment(x[1]))).toDF(["docId","segments"])

    featureFrame= ngramFrame.join(segments, "docId")

    featuresDF = featureFrame.rdd.map(lambda x: Row(did=x['docId'],docFeatures=x['opcodes']+x['2grams']+x['3grams']+x['4grams']+x['segments'])).toDF()

    featuresCV = featureFitModel.transform(featuresDF)

    testData = featuresCV.drop('docFeatures')
    testData.persist(StorageLevel(True, True, False, False, 1))
    saveData(testData,path)
    testData.show()

OpcodesList = sc.textFile("gs://dspp1/allOpcodes.txt")

opcodeSet = set(OpcodesList.collect())

opcodes= sc.broadcast(opcodeSet)

#train data

hashFiles = sc.textFile("gs://uga-dsp/project2/files/X_train.txt")
labels = sc.textFile("gs://uga-dsp/project2/files/y_train.txt")
hashFiles2 = sc.textFile("gs://uga-dsp/project2/files/X_test.txt")

featureFitModel = train(opcodes,labels,hashFiles,sc,sqlc,"gs://dspp1/final_large_train_features")

#test Data

test(opcodes,hashFiles2,sc,sqlc,"gs://dspp1/final_large_test_features",featureFitModel)
