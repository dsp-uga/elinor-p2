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
    parser.add_argument("-o", "--output", required = True,
        help = "The path to put the output ngram rdd.")
    args = vars(parser.parse_args())

    DATA_PATH  = "data/" + args['input']
    OUT_PATH   = "data/NGrams/" + args['output']
    
    spark = SparkSession\
        .builder\
        .appName("2Grammer")\
        .getOrCreate()

    sc = spark.sparkContext
    
    X1 = sc.textFile(DATA_PATH)\
              .flatMap(lambda x: x.split(" "))\
              .zipWithIndex()\
              .map(lambda x: swap(x))
    X2 = sc.textFile(DATA_PATH)\
              .flatMap(lambda x: x.split(" "))\
              .zipWithIndex()\
              .mapValues(lambda x: x-1)\
              .map(lambda x: swap(x))
  
    GRAMS = X1.join(X2)
    GRAMS.saveAsPickleFile(OUT_PATH)

    
                    
