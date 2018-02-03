'''
This is the softmax regression trainer that I wrote for project1. 
I have it written to work on the data from project one, 
and we might have to change it slightly to make it work better for project two.
Specifically, the preprocessing will either be different or unnecessary, 
the identifiers might need to be changed to be more reflective of the current project
(thought this might not actually be necessary), and - MOST IMPORTANTLY -
the classifier currently hits a wall in memory and run time around iteration 10. 

Here's the URL for a post I was referred to on Stack Overflow which addresses the question, 
though I don't know how relevant it is to my particular problem: 

https://stackoverflow.com/questions/31659404/spark-iteration-time-increasing-exponentially-when-using-join

The arg parser requires input training data and input training labels as well as 
a name for a file to store the trained model in. 

Alternate CL arguments: 
    "-t" or "--testing"       takes two arguemts:
                                  1. the path to find the X_test data in
                                  2. the path to find the y_test data in
    "-p" or "--predictions"   the name of the file to store predictions for the test data in
    "-d" or "--documents"     an indicator flag for creating a document classifier (DEFAULT: False)
    "-s" or "--sophisticated" an indicator flag for using sophisticated convergence calculation (DEFAULT: False)
'''

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
    parser.add_argument("-i", "--inputData", required = True,
        help = "The path to find the input training data.")
    parser.add_argument("-l", "--inputLabels", required = True,
        help = "The path to find the input training labels.")
    parser.add_argument("-o", "--output", required = True,
        help = "The file name to save the model to.")
    #optional arguments

    parser.add_argument("-t", "--testing", nargs=2,
        help = "The paths to find the testing data and testing labels.")
    parser.add_argument("-p", "--predictions",
        help = "The file name to save the predictions to.")
    parser.add_argument("-d", "--documents", action='store_true',
        help = "The file name to save the predictions to.")
    parser.add_argument("-s", "--sophisticated", action='store_true',
        help = "The file name to save the predictions to.")
    
    args = vars(parser.parse_args())

    DATA_PATH  = args['inputData']
    LABEL_PATH = args['inputLabels']
                        
    MODEL_OUT_FILE = args['output']
    PREDS_OUT_FILE = args['predictions']
                        
    TEST_DATA_PATH,TEST_LABEL_PATH = getTestPaths(args['testing'])

    DOC_DATA      = args['documents']
    SOPHISTICATED = args['sophisticated']
    
    spark = SparkSession\
        .builder\
        .appName("Project0")\
        .getOrCreate()

    sc = spark.sparkContext


    X     = sc.textFile(DATA_PATH)
    y     = sc.textFile(LABEL_PATH)

    X.cache()
    y.cache()

    if TEST_DATA_PATH != None :
        Xtest = sc.textFile(TEST_DATA_PATH)
        ytest = sc.textFile(TEST_LABEL_PATH)

    print("TEST_DATA_PATH: ",TEST_DATA_PATH,
          ", TEST_LABEL_PATH: ",TEST_LABEL_PATH)

    X     = sc.textFile(DATA_PATH)
    Xtest = sc.textFile(TEST_DATA_PATH)
    y     = sc.textFile(LABEL_PATH)
    ytest = sc.textFile(TEST_LABEL_PATH)

    X.cache()
    Xtest.cache()
    y.cache()
    ytest.cache()

    if DOC_DATA :

        ################*****PREPROCESSING*****##################

        y = y.zipWithIndex()\
             .map(lambda x: (x[1],x[0]))\
             .flatMapValues(lambda x: x.split(","))\
             .filter(lambda x: not x[1].find("CAT")==-1)
        y.cache()

        #We want to filter out any documents that don't have one of our labels off the bat
        #   y  .keys()     => did NON-DISTINCT
        #  ''  .distinct() => did DISTINCT
        #  ''  .map(...)   => (did,1)    #  ** This is our "filter rdd" **

        filterRDD = y.keys()\
                     .distinct()\
                     .map(lambda x: (x,1))


        #  X   .zipWithIndex() => (doc,did)
        #  ''  .map(...)       => (did,doc)
        #  ''  .roj('')        => (did,(doc,1)) for did in other
        #                                   OR
        #                         (did,(None,1)) for did not in this
        #  ''  .filter(...)    => (did,(doc,1)) i.e. gets rid of unimportant documents
        #  ''  .mapValues(...) => (did,doc)
        X = X.zipWithIndex()\
                 .map(lambda x: swap(x))\
                 .rightOuterJoin(filterRDD)\
                 .filter(lambda x: not x[1][0]==None)\
                 .mapValues(lambda x: x[0])

        X.cache()
        Xtest = Xtest.zipWithIndex()\
                             .map(lambda x : (x[1],x[0]))\
                             .flatMapValues(lambda x : x.split())\
                             .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
                             .distinct()
        Xtest.cache()
        
        #For multinomial logistic regression we have to create a duplicate document for duplicate labels
        #  y   .keys()              => did values (not necessarilly unique)
        #  ''  .map(...)            => (did,1)
        #  ''  .reduceByKey(add)    => (did,count) the number of times each did appears in y
        #  ''  .filter(...)         => (did,count) for all did which appear multiple times in y


        repeats = y.keys()\
                           .map(lambda x: (x,1))\
                           .reduceByKey(add)\
                           .filter(lambda x: x[1]>1)\
                           .mapValues(lambda x: x-1)

        #  X   .join(repeats)       => (did,(doc,count)) for each did which appears multiple times in y
        #  ''  .flatMapValues(...)  => creates count-1 many copies of each (did,doc) pair in the above rdd
        #  X   .union(XX)           => (did,doc) where did is NO LONGER UNIQUE
        #  ''  .sort(...)           => (did,doc) in sorted order)
        #  ''  .map(
        XX = X.join(repeats)\
              .flatMapValues(lambda x: [x[0]]*(x[1]))\

        X  = X.union(XX)\
              .sortByKey()\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0][1]))\
              .flatMapValues(lambda x : x.split())\
              .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
              .filter(lambda x: len(x[1])>1)
        X.cache()

        y = y.sortByKey()\
                .zipWithIndex()\
                .map(lambda x: (x[1],x[0][1]))
        y.cache()

        CONVERGED = False
        m = X.keys().count()
        alpha = .1
        epsilon = .001

        #Get the word counts for each document
        #   X  .map(...) => ((did,wid),1)
        #  ''  .rbk(add) => ((did,wid),count)
        #  ''  .map(...) => (did,wid,count)
        counts = X.map(lambda x: (x,1))\
                    .reduceByKey(add)\
                    .map(lambda x: (x[0][0],x[0][1],x[1]))
        counts.cache()
        
        #The size of the vocabulary
        B = counts.map(lambda x: x[1])\
                    .distinct()\
                    .count()

        #add the bias word on...
        #  X   .keys()     => (wid) NOT UNIQUE
        #  ''  .distinct() => (wid) UNIQUE
        #  ''  .zwi()      => (wid,i)
        #  ''  .map(...)   =>  (i,wid)
        keys = X.keys()\
                .distinct()\
                .zipWithIndex()\
                .map(lambda x: swap(x))


        #create a biase word for each document
        #  sc  .par      => (B,...,B) (m many B words)
        #  ''  .zwi      => [(B,1),(B,2),...,(B,m)]
        #  ''  .map(...) => [(1,B),(2,B),...,(m,B)]
        biasWords = sc.parallelize(['B']*m)\
                        .zipWithIndex()\
                        .map(lambda x: swap(x))

        #create a one count for each bias word
        #  sc  .par      => [1,...,1] (m many ones)
        #  ''  .zwi      => [(1,1),...,(1,m)]
        #  ''  .map(...) => [(1,1),...,(m,1)]
        ones = sc.parallelize(np.ones(m))\
                    .zipWithIndex()\
                    .map(lambda x: swap(x))
        #Join the Bias words to the bias counts and to the documents, using the indices to join
        #  keys  .join(...) => (i,(did,B))
        #  ''    .join(...) => (i,((did,B),1))
        #  ''    .values()  => ((did,B),1)
        #  ''    .map(...)  => (did,B,1)
        bias = keys.join(biasWords)\
                        .join(ones)\
                        .values()\
                        .map(lambda x: (x[0][0],x[0][1],x[1]))

        #add the bias words into the counts data
        counts  = counts.union(bias)
        counts.cache()

        N = counts.map(lambda x: x[1]).distinct().count()

        
        ##### ***** REGULARIZE THE COUNTS ****####
        #First get the mean value for each word
        #  counts  .map(...)       => (wid,count)
        #  ''      .rbk(...)       => (wid,sum_wid(count)))
        #  ''      .mapValues(...) => (wid, mu_wid)
        mean = counts.map(lambda x: (x[1],x[2]))\
                     .reduceByKey(add)\
                     .mapValues(lambda x: x/B)

        #Now get the standard deviation for each word
        #  counts  .map(...)    => (wid,count)
        #  ''      .join(mean)  => (wid,(count,mu_wid))
        #  ''      .mapValues() => (wid, (count-mu_wid)^2)
        #  ''      .rbk()       => (wid, var_wid)
        #  ''      .mapValues() => (wid, sd_wid)
        sd   = counts.map(lambda x: (x[1],x[2]))\
                     .join(mean)\
                     .mapValues(lambda x: (x[0]-x[1])*(x[0]-x[1]))\
                     .reduceByKey(add)\
                     .mapValues(lambda x: np.sqrt(x))

        #Join together for easier manipulation
        #  mean  .join(sd) => (wid,(mu_wid,sd_wid))
        regs = mean.join(sd)

        #Now regularize the counts as count = (count-mu)/sd
        #  counts  .map(...)   => (wid,(did,count))
        #  ''      .join(regs) => (wid,((did,count),(mu_wid,sd_wid)))
        #  ''      .map(...)   =>
        counts = counts.map(lambda x: (x[1],(x[0],x[2])))\
                        .join(regs)\
                        .map(lambda x: (x[1][0][0],x[0],(x[1][0][1]-x[1][1][0])/x[1][1][1]))
        counts.cache()
        #we need the initial population for theta = (lab,wid,w_wid)
        labs  = y.values().distinct()
        words = X.values().distinct().union(sc.parallelize(['B']))
        zeros = sc.parallelize([0.0])

        #Building theta
        #  labs  .cart(...) => (lab,wid) for each lab/wid pair
        #  ''    .cart(...) => ((lab,wid),0) for each lab/wid pair
        #  ''    .map(...)  => (lab,wid,w_wid) where w_wid is initialized to zero
        theta = labs.cartesian(words)\
                    .cartesian(zeros)\
                    .map(lambda x: (x[0][0],x[0][1],x[1]))
        theta.cache()

        K = labs.count()

        #Create the array of weights that we'll save to in the future

        ######################*************START THE GRADIENT DESCENT*************######################
        i=0
        J = 0
        while not CONVERGED :
            print("i: ",i)
            #Get the dot product of every document with every row in theta
            #  theta  .map(...) => (wid,(lab,w_wid))
            #  ''     .join(...) => (wid,((lab,w_wid),(did,count)))
            #  ''     .map(...)  => ((did,lab),w_wid*count)
            #  ''     .rbk(...)  => ((did,lab),x^(did)*DOT*theta^(lab))

            dots = theta.map(lambda x: (x[1],(x[0],x[2])))\
                        .join(counts.map(lambda x: (x[1],(x[0],x[2]))))\
                        .map(lambda x: ((x[1][1][0],x[1][0][0]),(x[1][0][1]*x[1][1][1])))\
                        .reduceByKey(add)
            dots.cache()

            #Now find the denominator for the predictions
            #  dots  .map(...) => (did,x^(did)*DOT*theta^lab) for each row in theta
            #  ''    .mapValues(...) => (did,exp(x^(did)*DOT*theta^lab))
            #  ''    .rbk(add)       => (did,Sum_lab(x^(did)*DOT*theta^lab))
            denom = dots.map(lambda x: (x[0][0],x[1]))\
                        .mapValues(lambda x: np.exp(x))\
                        .reduceByKey(add)

            #Now make our predictions
            #  dots  .map(...)  => (did,(lab,x^(did)*DOT*theta^lab)
            #  ''    .join(...) => (did,((lab,x^(did)*DOT*theta^lab),denom))
            #  ''    .map(...)  => ((did,lab),exp(x^(did)*DOT*theta^lab)/denom)
            h = dots.map(lambda x: (x[0][0],(x[0][1],x[1])))\
                    .join(denom)\
                    .map(lambda x: ((x[0],x[1][0][0]),(np.exp(x[1][0][1])/x[1][1])))
            h.cache()
            
            if False: #i%10 == 0 :
                # Calculate the loss
                # we need ((did,lab),1) pairs for each did and each lab
                yc  = y.keys()\
                        .distinct()\
                        .cartesian(y.values().distinct())\
                        .map(lambda x: (x,1))

                # now we'll get ((did,lab),1) pairs for each (did,lab) in y
                ycc = y.map(lambda x: (x,1))

                # Now we'll use the two to calculate the loss
                #  yc  .loj(ycc) =>  ((did,lab),(1,1)) when (did,lab) in ycc
                #                                     OR
                #                    ((did,lab),(1,None)) when (did,lab) not in ycc
                #  ''  .mapValues(...) => ((did,lab),1{y^did=lab})
                #  ''  .join(h...)     => ((did,lab),(1{y^did=lab},log(h(did,lab);theta)))
                #  ''  .mapValues(...) => ((did,lab),(-1/m * 1{y^did=lab}*log(h(did,lab);theta)))
                #  ''  .values()       => -1/m*1{y^did=lab}*log(h(did,lab);theta) FOR EACH did,lab combo
                #  ''  .reduce(add)    => -1/m*sum_did[sum_lab[1{y^did=lab}*log(h(did,lab);theta)]]
                J =  yc.leftOuterJoin(ycc)\
                        .mapValues(lambda x: 1 if x[1]==1 else 0)\
                        .join(h.mapValues(lambda x: np.log(x)))\
                        .mapValues(lambda x: x[0]*x[1]*-1/m)\
                        .values()\
                        .reduce(add)
                print("Cost:",J)
            i+=1
            #Now calculate the gradient
            #  y   .map(...)       => ((did,lab),1)
            #  h   .loj(yy)        => ((did,lab),(pred,1)) for (did,lab) in yy
            #                                    OR
            #                         ((did,lab),(pred,None)) for (did,lab) not in yy
            #  ''  .mapValues(...) => ((did,lab),1{y^did=lab}-pred) **NOTE: We'll refer to
            #                                                          1{y^did=lab}-pred
            #                                                         as expr from now on**
            #  ''  .map(...)       => (did,(lab,expr))
            #  ''  .join(counts)   => (did,((lab,expr),(wid,count)))
            #  ''  .map(...)       => ((lab,wid),expr*count)
            #  ''  .rbk(add)       => ((lab,wid),sum_did(expr*count))
            #  ''  .mapValues(...) => ((lab,wid),-1/m * sum_did(expr*count))
            yy = y.map(lambda x: (x,1))

            grad = h.leftOuterJoin(yy)\
                    .mapValues(lambda x: 1-x[0] if x[1]==1 else 0-x[0])\
                    .map(lambda x: (x[0][0],(x[0][1],x[1])))\
                    .join(counts.map(lambda x: (x[0],(x[1],x[2]))))\
                    .map(lambda x: ((x[1][0][0],x[1][1][0]),x[1][0][1]*x[1][1][1]))\
                    .reduceByKey(add)\
                    .mapValues(lambda x: -1/m * x )
            grad.cache()
            
            #find the new theta value
            # theta  .map(...)     => ((lab,wid),oldW)
            #  ''    .join(grad)   => ((lab,wid),(oldW,gradW))
            #  ''     .mapValues() => ((lab,wid),new_weight)
            thetaNew = theta.map(lambda x: ((x[0],x[1]),x[2]) )\
                            .join(grad)\
                            .mapValues(lambda x: x[0]-alpha*x[1])\
                            .map(lambda x: (x[0][0],x[0][1],x[1]))

            theta = thetaNew
            theta.cache()
            
            ##see if we need to stop updating anything...
            # theta  .map(...)  => ((lab,wid),w_wid)
            #  ''     .join(...) => ((lab,wid),(w_wid,new_weight))
            thetas = theta.map(lambda x: ((x[0],x[1]),x[2]))\
                        .join(thetaNew)

            if SOPHISTICATED:
                # We'll find the euclidean distance from the old weight vector
                # to the new weight vector for each label
                #  ''     .mapValues(...) => ((lab,wid),(w_wid - new_weight)^2)
                #                                          **NOTE**:    we'll refer to
                #                                                    (w_wid - new_weight)^2
                #                                                            as
                #                                                         diff_sqd
                #                                                       from here on
                #  ''     .map(...)       => (lab,diff_sqd)
                #  ''     .rbk(add)       => (lab,ssd)
                #  ''     .mapValues(...) => (lab,dist_euclid)

                dist = thetas .mapValues(lambda x: (x[0]-x[1])**2)\
                              .map(lambda x: (x[0][0],x[1]))\
                              .reduceByKey(add)\
                              .mapValues(lambda x: np.sqrt(x))
                # If each of the labels has converged according to epsilon, we'll call it a day
                # Otherwise we'll change the theta values that need to be changed
                #  thetas  .map(...) => (lab,(wid,(w_wid,new_weight)))
                #  ''      .join(dist) => (lab,((wid,(w_wid,new_weight)),dist))
                #  ''      .map(...)   => (lab,wid,w_wid) if dist<=epsilon
                #                                        OR
                #  ''                     (lab,wid,new_weight) if dist>epsilon
                
                if dist.values().map(lambda x: 1 if x>epsilon else 0).reduce(add) == 0 : CONVERGED = True
                else :
                    theta = thetas.map(lambda x: (x[0][0],(x[0][1],x[1])))\
                                  .join(dist)\
                                  .map(lambda x: (x[0],x[1][0][0],x[1][0][1][0])\
                                       if x[1][1] <= epsilon\
                                       else (x[0],x[1][0][0],x[1][0][1][1]) )
            #end sophisticated convergence testing        
            else :
                #The naive convergence
                if i == 200 : CONVERGED = True
        #end while
        with open(MODEL_OUT_FILE,"w") as f:
            theta.saveAsPickleFile(f)
    else :
        #TODO : refactor the classifier for the malware data
        print("TODO: FINISH")
        with open(MODEL_OUT_FILE,"w") as f:
            theta.saveAsPickleFile(f)
    
