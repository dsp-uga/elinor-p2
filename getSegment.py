
import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
import re


# In[2]:


sc = pyspark.SparkContext(appName="DocClassification")
sqlc = SQLContext(sc)


# In[3]:

labels = sc.textFile("gs://uga-dsp/project2/files/y_small_train.txt")

hashFiles = sc.textFile("gs://uga-dsp/project2/files/X_small_train.txt").map(lambda x: "gs://uga-dsp/project2/data/asm/"+ x+".asm")


# In[4]:


def fun(accum,x):
    return accum+','+x
HashFileString = hashFiles.reduce(fun)


# In[11]:


rdd1= sc.wholeTextFiles(HashFileString,20)



labelRdd = labels.zipWithIndex().map(lambda x: (x[1],x[0]))


labelFrame = labelRdd.toDF(["did","label"])

# # Extract Segments

# In[177]:

segments = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].splitlines())).map(lambda x: (x[0],[re.findall(r'\w+:?(?=:)',word) for word in x[1]])).flatMapValues(lambda x: x).map(lambda x: (x[0],x[1][0])).map(lambda x: (x,1)).reduceByKey(add)


segmentsRdd = segments.reduceByKey(add).map(lambda x: (x[0][1],(x[0][0],x[1])))


vocab = segmentsRdd.keys().distinct().zipWithIndex()


segmentsRdd = segmentsRdd.join(vocab).map(lambda x: (x[1][0][0],x[1][1],float(x[1][0][1])))


segmentsRdd2 = segmentsRdd.map(lambda x: MatrixEntry(x[0],x[1],float(x[2])))
matSegment = CoordinateMatrix(segmentsRdd2).toIndexedRowMatrix().rows.toDF(["did","features"])


finSegment = matSegment.join(labelFrame,['did'])


finSegment.write.parquet("gs://elinor-p2/SegmentOutput")
