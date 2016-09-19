cs224d:Deep Learning For Natural Language Process

My own solution for 3 problem set of cs224d. contact with me at 158085159@qq.com

Course Description

Natural language processing (NLP) is one of the most important technologies of the information age.  <br/>
Understanding complex language utterances is also a crucial part of artificial intelligence. Applications of NLP are everywhere because people communicate most everything in language: web search, advertisement, emails, customer service, language translation, radiology reports, etc. There are a large variety of underlying tasks and machine learning models powering NLP applications. Recently, deep learning approaches have obtained very high performance across many different NLP tasks. These models can often be trained with a single end-to-end model and do not require traditional, task-specific feature engineering. In this spring quarter course students will learn to implement, train, debug, visualize and invent their own neural network models. The course provides a deep excursion into cutting-edge research in deep learning applied to NLP. The final project will involve training a complex recurrent neural network and applying it to a large scale NLP problem. On the model side we will cover word vector representations, window-based neural networks, recurrent neural networks, long-short-term-memory models, recursive neural networks, convolutional neural networks as well as some very novel models involving a memory component. Through lectures and programming assignments students will learn the necessary engineering tricks for making neural networks work on practical problems.

=============================================================================================================
PSet1: 情感分析  <br/>
Q1: Softmax (10 points)<br/>

Q2: Neural Network Basics 神经网络基础(30 points) <br/>
h = sigmoid(xW1 + b1) , y^ = softmax(hW2 + b2) <br/>

Q3: word2vec 词向量(40 points + 5 bonus)<br/>
skipgram,negative sampling,CBOW  <br/>

Q4: Sentiment Analysis 情感分析 (20 points) <br/>
Process:<br/>
1)word embeding---><br/>
2)average word vector to get features(X)---><br/>
3)dot product with Learnable Weight(W)----><br/>
4)Sofmtax to get possibility<br/>
5)calculate Loss, and Gradient, update W <br/>
with the word vectors you trained, we are going to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we’ll just use ﬁve classes:<br/>
“very negative”, “negative”, “neutral”, “positive”, “very positive”


=============================================================================================================
PSet2: 命名实体识别&& 循环神经网络下预测下一个词<br/>
1 Tensorﬂow Softmax (20 points)<br/> 
2 Deep Networks for Named Entity Recognition 命名实体识别 (35 points) <br/>
get to practice backpropagation and training deep networks to attack the task of Named Entity Recognition: predicting whether a given word, in context, represents one of four categories: • Person (PER) • Organization (ORG) • Location (LOC) • Miscellaneous (MISC)
Process:<br/>
1)word embeding to get word vectors 词向量化----><br/>
                                    上线文（将左右两个边的词连接）<br/>
2) represent context as a “window” consisting of a word concatenated with its immediate neighbors：<br/>
    x(t) = [xt−1L, xtL, xt+1L] ∈R3d <br/>
3)  1-hidden-layer neural network   神经网络<br/>
    h = tanh(x(t)W + b1) <br/>
4)  compute our prediction: softmax 线性计算后的softmax值<br/>
     y^=softmax(hU+b2)<br/>
5)  evaluate cross-entropy loss:    计算损失<br/>
     J=CE(y,y^)=sum(y*logy^)<br/>
6)  back-prop                       反向传播 <br/>

3 Recurrent Neural Networks: Language Modeling: 循环神经网络下预测下一个词 (45 points) <br/>
Given words x1,...,xt, a language model predicts the following word xt+1 by modeling: P(xt+1 = vj | xt,...,x1) <br/>
PROCESS:<br/>
implement a recurrent neural network language model, which uses feedback information in the hidden layer to model the “history” <br/> xt,xt−1,...,x1. Formally, the model is, for t = 1,...,n−1:
1) word embeding:                                           词向量化---><br/>
2) get next hidden state:                                   计算下一个隐藏层的状态(循环：从state0开始到final state)<br/>
      h(t) = sigmoid(h(t−1)*H + e*I + b1) <br/> 
3) get possibilities after linear layer and compute softmax  线性运算后计算softmax后的概率<br/>
      y^(t) = softmax(h(t)U + b2) <br/>
4) compute sequence loss:                                    计算序列的损失 <br/>
5) back-prop                                                 反向传播 <br/>


=============================================================================================================
PSet3:  Recursive Neural Network：递归神经网络下的情感分析  <br/>
sentiment classes: Really Negative, Negative, Neutral, Positive, and Really Positive  <br/>
PROCESS:<br/>
1)  Concatenate node from left and right,linear layer,Relu layer.使用树形结构底层左右两个节点的线性运算，再计算RELU值(迭代)<br/>
h = max([ h_Left,h_Right]*W + b,0) <br/>
2) Linear layer, Softmax                                         线性运算再计算概率<br/>
y = softmax(h*U + b)<br/>
3) Compute loss                                                  计算损失<br/>
4) Back-prop                                                     反向传播<br/>
