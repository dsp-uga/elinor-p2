import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
import re

sc = pyspark.SparkContext(appName="DocClassification")
sqlc = SQLContext(sc)

hashFiles = sc.textFile("gs://uga-dsp/project2/files/X_small_train.txt")
asmFiles = hashFiles.map(lambda x: "gs://uga-dsp/project2/data/asm/"+ x+".asm")

labels = sc.textFile("gs://uga-dsp/project2/files/y_small_train.txt")

def fun(accum,x):
    return accum+','+x
asmFileString = asmFiles.reduce(fun)


rdd1= sc.wholeTextFiles(asmFileString,20)

OpcodesList = sc.textFile("gs://elinor-p2/allOpcodes.txt")
opcodes= sc.broadcast(OpcodesList.collect())


opcodesInDoc = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in opcodes.value]).zipWithIndex().map(lambda x: (x[1],x[0]))


ngramFrame = sqlc.createDataFrame(opcodesInDoc,["docId","opcodes"])


twoGram = NGram(n=2, inputCol="opcodes", outputCol="2grams")
ngramFrame = twoGram.transform(ngramFrame)


threeGram = NGram(n=3, inputCol="opcodes", outputCol="3grams")
ngramFrame= threeGram.transform(ngramFrame)


fourGram = NGram(n=4, inputCol="opcodes", outputCol="4grams")
ngramFrame = fourGram.transform(ngramFrame)


twoGramRdd = ngramFrame.select("docId","2grams").rdd.map(tuple)
threeGramRdd =ngramFrame.select("docId","3grams").rdd.map(tuple)
fourGramRdd =ngramFrame.select("docId","4grams").rdd.map(tuple)

oneGramCounts = opcodesInDoc.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


twoGramCounts = twoGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)

threeGramCounts = threeGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


fourGramCounts = fourGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


segments = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].splitlines())).map(lambda x: (x[0],[re.findall(r'\w+:?(?=:)',word) for word in x[1]])).flatMapValues(lambda x: x).map(lambda x: (x[0],x[1][0])).map(lambda x: (x,1)).reduceByKey(add)





labelRdd = labels.zipWithIndex().map(lambda x: (x[1],x[0]))

labelFrame = labelRdd.toDF(["did","label"])


allFeatures = sc.union([oneGramCounts,twoGramCounts,threeGramCounts,fourGramCounts,segments])


allFeatures = allFeatures.reduceByKey(add).map(lambda x: (x[0][1],(x[0][0],x[1])))


vocab = allFeatures.map(lambda x: (x[0],1)).reduceByKey(add).map(lambda x: x[0]).zipWithIndex()


allFeaturesJoined = allFeatures.join(vocab).map(lambda x: (x[1][0][0],x[1][1],x[1][0][1]))


allFeatureMat = allFeaturesJoined.map(lambda x: MatrixEntry(x[0],x[1],x[2]))
mat = CoordinateMatrix(allFeatureMat).toIndexedRowMatrix().rows.toDF(["did","features"])



fin = mat.join(labelFrame,['did'])

fin.write.parquet("gs://elinor-p2/train_features")
