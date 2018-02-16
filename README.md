# elinor-p2
Given several `.bytes` and `.asm` files for several  the current project aims to classify documents from 1 of 9 groups. Each document can only have one class. To this end, we conducted research and derived inspiration from Kaggle competition winners that this project is based on. In their presentation, they note several features to be advantages to them:
  - Unigram through 4grams for `bytes` files.
  - Unigram through 4grams for `opcodes` in `asm` files. (`opcodes` are translations of bytes files that correspond to specific operation
  - Denoted segments in `asm` files such as the `HEADER`

We trained our features on the `pyspark` `ml.classifaction.RandomForestClassifier` starting the default settings of the classifier, and then tweaking to try upto 30 trees with max depth of 8.

Due to memory issues we were not able to use all the features initially. Thus we limited ourselves to subsets of these features and tried various combinations of subsets. 

Inspite of this work-around and also tweaking the classifiers in multiple ways we were not able to exceed 27.6% accuracy on the final test set.
