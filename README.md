cs224d:Deep Learning For Natural Language Process

My own solution for 3 problem set of cs224d. contact with me at 158085159@qq.com

Course Description

Natural language processing (NLP) is one of the most important technologies of the information age. 
Understanding complex language utterances is also a crucial part of artificial intelligence. Applications of NLP are everywhere because people communicate most everything in language: web search, advertisement, emails, customer service, language translation, radiology reports, etc. There are a large variety of underlying tasks and machine learning models powering NLP applications. Recently, deep learning approaches have obtained very high performance across many different NLP tasks. These models can often be trained with a single end-to-end model and do not require traditional, task-specific feature engineering. In this spring quarter course students will learn to implement, train, debug, visualize and invent their own neural network models. The course provides a deep excursion into cutting-edge research in deep learning applied to NLP. The final project will involve training a complex recurrent neural network and applying it to a large scale NLP problem. On the model side we will cover word vector representations, window-based neural networks, recurrent neural networks, long-short-term-memory models, recursive neural networks, convolutional neural networks as well as some very novel models involving a memory component. Through lectures and programming assignments students will learn the necessary engineering tricks for making neural networks work on practical problems.

PSet1: Q1: Softmax (10 points)

Q2: Neural Network Basics 神经网络基础(30 points)
h = sigmoid(xW1 + b1) , y^ = softmax(hW2 + b2)

Q3: word2vec 词向量(40 points + 5 bonus)
skipgram,negative sampling,CBOW 

Q4: Sentiment Analysis 情感分析 (20 points) <br/>
Process:
1)word embeding---><br/>
2)average word vector to get features(X)---><br/>
3)dot product with Learnable Weight(W)----><br/>
4)Sofmtax to get possibility<br/>
5)calculate Loss, and Gradient, update W<br/>
with the word vectors you trained, we are going to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we’ll just use ﬁve classes:
“very negative”, “negative”, “neutral”, “positive”, “very positive”


PSet2:
1 Tensorﬂow Softmax (20 points) 
2 Deep Networks for Named Entity Recognition (35 points) 
3 Recurrent Neural Networks: Language Modeling (45 points)

PSet3: 1 RNN’s (Recursive Neural Network)
