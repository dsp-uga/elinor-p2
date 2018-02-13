
# coding: utf-8

# In[1]:


import findspark
findspark.init()
import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
import re


# In[2]:


sc = pyspark.SparkContext('local[*]',appName="DocClassification")
sqlc = SQLContext(sc)


# In[3]:


rdd1= sc.wholeTextFiles("/home/vyom/UGA/DSP/Project2/data/train/asm/0bN6ODYWw2xeCQBn3tEg.asm,/home/vyom/UGA/DSP/Project2/data/train/asm/0eN9lyQfwmTVk7C2ZoYp.asm,/home/vyom/UGA/DSP/Project2/data/train/asm/0H63jydvIahOVqgx5Kfo.asm,/home/vyom/UGA/DSP/Project2/data/train/asm/0hZEqJ5eMVjU21HAG7Ii.asm")
#rdd1= sc.wholeTextFiles("/home/vyom/UGA/DSP/Project2/data/train/asm/")


# In[4]:


#rdd1.count()


# In[4]:


OpcodesList = sc.textFile("/home/vyom/UGA/DSP/Project2/allOpcodes.txt")
opcodes= sc.broadcast(sc.textFile("/home/vyom/UGA/DSP/Project2/allOpcodes.txt").collect())


# # Get Opcodes list using python approach

# In[6]:


opcodesInDoc = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in opcodes.value]).zipWithIndex().map(lambda x: (x[1],x[0]))


# In[7]:


#opcodesInDoc.take(10)


# # Get Opcodes list using mostly spark

# In[8]:


#opcodesInDoc = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].split())).flatMapValues(lambda x: x).filter(lambda x: x[1] in opcodes.value).groupByKey().map(lambda x: (x[0],list(x[1])))


# In[9]:


#opcodesInDoc.take(10)


# # Get N-grams and N-grams count

# In[10]:


ngramFrame = sqlc.createDataFrame(opcodesInDoc,["docId","opcodes"])


# In[12]:


twoGram = NGram(n=2, inputCol="opcodes", outputCol="2grams")
ngramFrame = twoGram.transform(ngramFrame)


# In[13]:


threeGram = NGram(n=3, inputCol="opcodes", outputCol="3grams")
ngramFrame= threeGram.transform(ngramFrame)


# In[14]:


fourGram = NGram(n=4, inputCol="opcodes", outputCol="4grams")
ngramFrame = fourGram.transform(ngramFrame)


# In[15]:


twoGramRdd = ngramFrame.select("docId","2grams").rdd.map(tuple)
threeGramRdd =ngramFrame.select("docId","3grams").rdd.map(tuple)
fourGramRdd =ngramFrame.select("docId","4grams").rdd.map(tuple)


# In[16]:


oneGramCounts = opcodesInDoc.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: ((x[0][0],x[0][1]),x[1]))


# In[17]:


twoGramCounts = twoGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: ((x[0][0],x[0][1]),x[1]))


# In[18]:


threeGramCounts = threeGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: ((x[0][0],x[0][1]),x[1]))


# In[19]:


fourGramCounts = fourGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: ((x[0][0],x[0][1]),x[1]))


# # Get the sparse matrix

# In[21]:


labels = sc.textFile("/home/vyom/UGA/DSP/Project2/data/train/y_small_train.txt")


# In[22]:


labelRdd = labels.zipWithIndex()


# In[23]:


labelRdd = sc.parallelize(labelRdd.take(4))


# In[24]:


labelFrame = labelRdd.toDF(["did","label"])


# In[25]:


allFeatures = sc.union([oneGramCounts,twoGramCounts,threeGramCounts,fourGramCounts])


# In[26]:


allFeatures = allFeatures.reduceByKey(add).map(lambda x: (x[0][1],(x[0][0],x[1])))


# In[27]:


vocab = allFeatures.keys().distinct().zipWithIndex()


# In[28]:


allFeaturesJoined = allFeatures.join(vocab).map(lambda x: (x[1][0][0],x[1][1],x[1][0][1]))


# In[29]:


allFeatureMat = allFeaturesJoined.map(lambda x: MatrixEntry(x[0],x[1],x[2]))
mat = CoordinateMatrix(allFeatureMat).toIndexedRowMatrix().rows.toDF(["did","features"])


# In[30]:


mat.write.parquet("/home/vyom/UGA/DSP/Project2/output3")
