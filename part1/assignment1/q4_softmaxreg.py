import numpy as np
import random

from cs224d.data_utils import *

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params

def getSentenceFeature(tokens, wordVectors, sentence):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    # Implement computation for the sentence features given a sentence.                                                       
    
    # Inputs:                                                         
    # - tokens: a dictionary that maps words to their indices in  #e.g. tokens=['a':0,'i',:1,'like:2,....]  
    #          the word vector list                                
    # - wordVectors: word vectors (each row) for all tokens    #            
    # - sentence: a list of words in the sentence of interest  #e.g. 'i','like','paris','very','much'

    # Output:                                                         
    # - sentVector: feature vector for the sentence    
    sentVector = np.zeros((wordVectors.shape[1],))
    
    ### YOUR CODE HERE
    sentence_length=len(sentence)
    #print "wordVectors:",wordVectors.shape,"sentVector:",sentVector.shape,"sentence:",sentence,"sentence_length:",sentence_length
    for i in xrange(sentence_length): #indices = [tokens[word] for word in sentence]
        word=sentence[i]  #get the word
        index=tokens[word]  #get the index of the word
        sentVector+=wordVectors[index] #get word vector by index, then sum up
    
    sentVector=sentVector/sentence_length #sentVector = np.mean(wordVectors[indices, :], axis=0)
    #print "sentVecotor:",sentVector
    #raise NotImplementedError
    ### END YOUR CODE
    
    return sentVector

def softmaxRegression(features, labels, weights, regularization = 0.0, nopredictions = False):
    """ Softmax Regression """
    # Implement softmax regression with weight regularization.        
    
    # Inputs:                                                         
    # - features: feature vectors, each row is a feature vector  #X   
    # - labels: labels corresponding to the feature vectors      #y   
    # - weights: weights of the regressor                        #W  
    # - regularization: L2 regularization constant                  
    
    # Output:                                                         
    # - cost: cost of the regressor                                 
    # - grad: gradient of the regressor cost with respect to its    
    #        weights                                               
    # - pred: label predictions of the regressor (you might find    
    #        np.argmax helpful)  
    #print "features:",features.shape,"labels:",labels,"weights.shape:",weights.shape #features: (10L, 10L) labels: (10L,)  labels:<type 'numpy.ndarray'>
    prob = softmax(features.dot(weights))
    #print "prob:",prob
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N
    ######################################################################
    labels2=np.zeros((N,weights.shape[1]))
    labels_list=list(labels)
    labels2[(xrange(N),labels_list)]=1
    cost2=-1.0/N*np.sum(labels2*np.log(prob))
    ######################################################################
    cost += 0.5 * regularization * np.sum(weights ** 2)
    
    ### YOUR CODE HERE: compute the gradients and predictions
    #print "prob.shape:",prob.shape,"labels2.shape:",labels2.shape,"weights.shape:",weights.shape
    grad = np.array(prob)
    grad[range(N), labels] -= 1.0
    grad = features.T.dot(grad) / N
    grad += regularization * weights
    #print "grad:",grad
    #grad2=features.dot(prob-labels2)/N+regularization*weights
    #print "grad2:",grad2
    if N>1:
        pred=np.argmax(prob,axis=1) #find the index for prediction.
    else:
        pred=np.argmax(prob)
    
        
    #raise NotImplementedError
    ### END YOUR CODE
    
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization = 0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights, 
        regularization)
    return cost, grad

def sanity_check():
    """
    Run python q4_softmaxreg.py.
    """
    random.seed(314159)
    np.random.seed(265)

    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
    dimVectors = wordVectors.shape[1]

    dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
    dummy_features = np.zeros((10, dimVectors))
    dummy_labels = np.zeros((10,), dtype=np.int32)    
    for i in xrange(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
    print "==== Gradient check for softmax regression ===="
    gradcheck_naive(lambda weights: softmaxRegression(dummy_features,
        dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

    print "\n=== Results ==="
    print softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0)

if __name__ == "__main__":
    sanity_check()