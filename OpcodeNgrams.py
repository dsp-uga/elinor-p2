
# coding: utf-8

# In[33]:


import pyspark
from operator import add
from pyspark import SparkConf
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col,udf
from pyspark.sql import SQLContext


# In[2]:


sc = pyspark.SparkContext('local[*]',appName="DocClassification")
sqlc = SQLContext(sc)


# In[3]:


rdd1= sc.wholeTextFiles("/home/vyom/UGA/DSP/Project2/data/train/asm/0bN6ODYWw2xeCQBn3tEg.asm,/home/vyom/UGA/DSP/Project2/data/train/asm/0eN9lyQfwmTVk7C2ZoYp.asm")


# In[5]:


opcodes= sc.broadcast(sc.textFile("/home/vyom/UGA/DSP/Project2/allOpcodes.txt").collect())


"""
# # Get Opcodes list using python approach

# In[7]:


rdd2 = rdd1.map(lambda x: x[1].split()).map(lambda x: [word for word in x if word in opcodes.value])


# In[8]:


rdd2.collect()


# # Get Opcodes list using mostly spark
"""
# In[21]:


opcodesInDoc = rdd1.zipWithIndex().map(lambda x: (x[1],x[0][1].split())).flatMapValues(lambda x: x).filter(lambda x: x[1] in opcodes.value).groupByKey().map(lambda x: (x[0],list(x[1])))



# # Get N-grams

# In[77]:


ngramFrame = sqlc.createDataFrame(opcodesInDoc,["docId","opcodes"])


# In[78]:


twoGram = NGram(n=2, inputCol="opcodes", outputCol="2grams")
ngramFrame = twoGram.transform(ngramFrame)


# In[79]:


threeGram = NGram(n=3, inputCol="opcodes", outputCol="3grams")
ngramFrame = threeGram.transform(ngramFrame)


# In[80]:


fourGram = NGram(n=4, inputCol="opcodes", outputCol="4grams")
ngramFrame = fourGram.transform(ngramFrame)


# In[91]:


twoGramRdd = ngramFrame.select("docId","2grams").rdd.map(tuple)
threeGramRdd =ngramFrame.select("docId","3grams").rdd.map(tuple)
fourGramRdd =ngramFrame.select("docId","4grams").rdd.map(tuple)


# In[158]:


oneGramCounts = opcodesInDoc.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


# In[153]:


twoGramCounts = twoGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


# In[154]:


threeGramCounts = threeGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


# In[155]:


fourGramCounts = fourGramRdd.flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)


# In[174]:


print(oneGramCounts.take(10))

