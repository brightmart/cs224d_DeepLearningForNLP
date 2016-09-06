import numpy as np
import matplotlib.pyplot as plt

from cs224d.data_utils import *

from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
REGULARIZATION = None   # Assign a list of floats in the block below
### YOUR CODE HERE
#raise NotImplementedError
REGULARIZATION=[0.00001,0.0001,0.001]
### END YOUR CODE

# 1.Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# 2.Load the word vectors we trained earlier 
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:]) #add two vectors up to get 'One' vector as word vector.
dimVectors = wordVectors.shape[1]

# 3.Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words) # Obtain the sentence feature for sentiment analysis by averaging its word vectors( Implement computation for the sentence features given a sentence.)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

###########################################################Add by bright: test set features
testset=dataset.getTestSentences()
nTest=len(testset)
testFeatures=np.zeros((nTest,dimVectors))
testLabels=np.zeros((nTest,),dtype=np.int32)
for i in xrange(nTest):
    words,testLabels[i]=testset[i]
    testFeatures[i,:]=getSentenceFeature(tokens,wordVectors,words)
#########################################################Add by bright: test set features
# Try our regularization parameters
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print "Training for reg=%f" % regularization 

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, 
        weights, regularization), weights, 3.0, 10000, PRINT_EVERY=2000) #PRINT_EVERY=100--->10000

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print "Train accuracy (%%): %f" % trainAccuracy

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print "Dev accuracy (%%): %f" % devAccuracy

    # Save the results and weights
    results.append({
        "reg" : regularization, 
        "weights" : weights, 
        "train" : trainAccuracy, 
        "dev" : devAccuracy})

# Print the accuracies
print ""
print "=== Recap ==="
print "Reg\t\tTrain\t\tDev"
for result in results:
    print "%E\t%f\t%f" % (
        result["reg"], 
        result["train"], 
        result["dev"])
print ""

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None

### YOUR CODE HERE 
#results=[{'reg':0.1,'weights':0.5,'train':0.3,'dev':0.3},{'reg':0.01,'weights':0.51,'train':0.4,'dev':0.4}]
BEST_REGULARIZATION=-1000
BEST_WEIGHTS=None
BEST_DEV_ACCURACY=-1
for result in results:
    if result['dev']>BEST_DEV_ACCURACY:
        BEST_DEV_ACCURACY=result['dev']
        BEST_REGULARIZATION=result['reg']
        BEST_WEIGHTS=result['weights']
    
#raise NotImplementedError
### END YOUR CODE

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print "Best regularization value: %E" % BEST_REGULARIZATION
print "Test accuracy (%%): %f" % accuracy(testLabels, pred)

# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("q4_reg_v_acc.png")
plt.show()

