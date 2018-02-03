from pyspark.sql import SparkSession
from operator import add
import argparse
import numpy as np
import json
import string

def getTestPaths(args):
    if args==None: return (None,None)
    else :
        return (args[0],args[1])


def swap(tup):
    return (tup[1],tup[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Softmax Classifier for Project 2 CSCI 8370  DSP18",
        epilog = "A Softmax Regression Trainer for Document Classification",
        add_help = "How to use",
        prog = "python SMTrainer.py -i <input_data_file> -l <input_labal_file> -o <output_directory> [optional args]")

    #required arguments
    parser.add_argument("-i", "--input", required = True,
        help = "The path to find the document to be n-grammed.")
    parser.add_argument("-n", "--gramSize", required = True,
        help = "The size of the n-grams to make.")
    parser.add_argument("-o", "--output", required = True,
        help = "The name of the file to save the n-grams to.")
    
    args = vars(parser.parse_args())

    DATA_PATH  = args['inputData']
    N          = args['gramSize']
    OUT_PATH   = "data/NGrams/" + args['output']
    
    spark = SparkSession\
        .builder\
        .appName("Project0")\
        .getOrCreate()

    sc = spark.sparkContext

    RDDs = []

    for i in np.arange(N):
        X = sc.textFile(DATA_PATH)\
              .flatMap(lambda x: x.split(" "))\
              .map(lambda x: (x,1-i))\
              .map(lambda x: swap(x))
        RDDs += [X]

    X = RDDs[0]

    for i in np.arange(N-1)+1:
        X = X.join(RDDs[i])\
             .flatMapValues(lambda x: x)

    X.saveAsPickleFile(OUT_PATH)

    
                    
