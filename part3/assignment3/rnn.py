import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab

#Recursive Nenural Net: Positive/Negative Sentiment Analysis of Sentences using Standford Sentiment Dataset
RESET_AFTER = 50
class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 35
    label_size = 2
    output_size=label_size ######################add by Bright. 2016.09.03
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 1 #30 ######################################################################################################
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)


class RNN_Model():

    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(700, 100, 200) #load train/dev/test data

        # build vocab from training data
        self.vocab = Vocab() #get a Vocab object with many functions.e.g. construct,add...
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents))) #calling construct of vocab to add sentences. # chain('ABC', 'DEF') --> A B C D E F

    def inference(self, tree, predict_only_root=False):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.iteritems() if node.label!=2]
            node_tensors = tf.concat(0, node_tensors) #t1 = [[1, 2, 3], [4, 5, 6]] t2 = [[7, 8, 9], [10, 11, 12]] ======> tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        return self.add_projections(node_tensors) #e.g. like [3.3,4.0,1.0,15.0,....]

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        '''
        with tf.variable_scope('Composition'):
            ### YOUR CODE HERE
            #pass #################################################################################################1.
            tf.get_variable("embedding",[self.vocab.total_words, self.config.embed_size])
            tf.get_variable("W1",[2*self.config.embed_size,self.config.embed_size])
            tf.get_variable("b1",[1,self.config.embed_size])
            ### END YOUR CODE
        with tf.variable_scope('Projection'):
            ### YOUR CODE HERE
            #pass ##################################################################################################2
            tf.get_variable("U",[self.config.embed_size,self.config.label_size])
            tf.get_variable("bs",[1,self.config.label_size])   
            ### END YOUR CODE

    def add_model(self, node):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to 
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node_tensor so it can be   # node.tensor--------> node_tensor
              used by it's parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        with tf.variable_scope('Composition', reuse=True):
            ### YOUR CODE HERE
            #pass #################################################################################################3.ADD MODEL IS IMPORTANT PART.
            embedding=tf.get_variable("embedding")
            W1=tf.get_variable("W1")
            b1=tf.get_variable("b1")
            #node_left_right_=[node.left,node.right]
            #node_left_right=tf.convert_to_tensor(node_left_right_)
            ### END YOUR CODE


        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf: #it's vector representation is just that of the word vector (see tf.gather())
            ### YOUR CODE HERE
            #pass #################################################################################################4.
            #curr_node_tensor=tf.nn.embedding_lookup(embedding,node)
            word_id = self.vocab.encode(node.word) #word_to_index              tf.expand_dims(input, dim)----->Inserts a dimension of 1 into a tensor's shape.
			                                                                 # 't' is a tensor of shape [2].shape(expand_dims(t, 0)) ==> [1, 2]
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, word_id), 0) #tf.gather(params, indices)--->Gather slices from params according to indices
            #tf.gather(node) 
            ### END YOUR CODE
        else: #it is not a leaf
            node_tensors.update(self.add_model(node.left)) #operation:recursive dict0.update---->add key-value pairs to dict0
            node_tensors.update(self.add_model(node.right))#operation:recursive 
            ### YOUR CODE HERE
            #pass #################################################################################################5.
                                                         #t1 = [[1, 2, 3], [4, 5, 6]] t2 = [[7, 8, 9], [10, 11, 12]] ======> tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            child_tensor = tf.concat(1, [node_tensors[node.left], node_tensors[node.right]])            
            z=tf.matmul(child_tensor,W1)+b1 
            curr_node_tensor=tf.nn.relu(z,0) 
            ### END YOUR CODE
        node_tensors[node] = curr_node_tensor #add value to dict-----> value('curr_node_tensor') to dict('node_tensors') with key('node')
        return node_tensors

    def add_projections(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        ### YOUR CODE HERE
        #pass #################################################################################################6.
        with tf.variable_scope("Projection",reuse=True):
            U=tf.get_variable("U")
            bs=tf.get_variable("bs")
            logits=tf.matmul(node_tensors,U)+bs
        ### END YOUR CODE
        return logits

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        loss = None
        # YOUR CODE HERE
        #pass #################################################################################################7
        #1.get and add data loss
        loss_data=tf.nn.sparse_softmax_cross_entropy_with_logits(labels,logits) #calling functions of softmax,cross entropy loss by providing label and un-flattenned scores
        loss_data=tf.reduce_mean(loss_data) #get a single number by taking mean of the data.
        #tf.add_to_collection('total_loss',loss_data) #add the loss to total loss.
        #2.get and add regularization loss
        with tf.variable_scope('Composition', reuse=True):
            W1=tf.get_variable("W1")
            #b1=tf.get_variable("b1")
        with tf.variable_scope('Projection', reuse=True):
            U=tf.get_variable("U")
            #bs=tf.get_variable("bs")
        loss_regularization=self.config.l2*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(U))
        #tf.add_to_collection('total_loss',loss_regularization)
        #loss=tf.add_n(tf.get_collection('total_loss'))
        loss=loss_data+loss_regularization
        # END YOUR CODE
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        train_op = None
        # YOUR CODE HERE
        #pass #################################################################################################8.
        optimizer=tf.train.GradientDescentOptimizer(self.config.lr)
        #global_step=tf.Variable(0,name='global_step',trainable=False)
        train_op=optimizer.minimize(loss) #,global_step=global_step
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        #pass #################################################################################################9.
        print "y:",y
        predictions=tf.argmax(y, 1) #each row get a index which element has highest value.
        # END YOUR CODE
        return predictions

    def __init__(self, config): #when init, assign config, and load data to RNN
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        for i in xrange(int(math.ceil(len(trees)/float(RESET_AFTER)))): #math.ceil--->get the nearest int.
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree, True) #1.use tree to get raw score
                    predictions = self.predictions(logits) #2.use logit to get index of predicted label.
                    root_prediction = sess.run(predictions)[0] #3.evaluate the the index of predicted label.
                    if get_loss:
                        root_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label])) #4.calucate loss by using softmax,cross entropy, and regularization
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model = False, verbose=True): #Run one epoch
        step = 0
        loss_history = []
        #1.go though each traing data by taking forward pass,calcuate loss and update weights
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.initialize_all_variables()
                    sess.run(init)
                else:
                    saver = tf.train.Saver() #Saves and restores variables
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name) #Restore variable using saver
                for _ in xrange(RESET_AFTER):
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step] #1.get training tree
                    logits = self.inference(tree) #2.forward pass
                    labels = [l for l in tree.labels if l!=2]
                    loss = self.loss(logits, labels) #3.calucate loss
                    train_op = self.training(loss) #4.set up traing Ops
                    loss, _ = sess.run([loss, train_op]) #5.evaluate loss(also update weights)
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name)
                
        #2.calucate training accuracy, validation accuracy after each epoch
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name) 
        val_preds, val_losses = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print
        print 'Training acc (only root node): {}'.format(train_acc)
        print 'Valiation acc (only root node): {}'.format(val_acc)
        print self.make_conf(train_labels, train_preds)
        print self.make_conf(val_labels, val_preds)
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in xrange(self.config.max_epochs): #loop over max_epochs by calling run_epoch function.
            print 'epoch %d'%epoch
            #1.training one epoch.
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch()
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #2.lr annealing (optional, but it can improve perfomrance of model.)
            epoch_loss = np.mean(loss_history)
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold: #if epoch loss is not decreasing after one epoch, then decay 'learn rate' 
                self.config.lr/=self.config.anneal_by
                print 'annealed lr to %f'%self.config.lr
            prev_epoch_loss = epoch_loss

            #3.save if model has improved on val
            if val_loss < best_val_loss:
                 shutil.copyfile('./weights/%s.temp'%self.config.model_name, './weights/%s'%self.config.model_name) #copy file from source to target location.
                 best_val_loss = val_loss
                 best_val_epoch = epoch

            #4.if model has not imprvoved for a while stop (we can stop training by using 'break' when meet condition)
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        print '\n\nstopped at %d\n'%stopped
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in itertools.izip(labels, predictions):
            confmat[l, p] += 1
        return confmat


def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Sentiment Classification.
    When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print 'Training time: {}'.format(time.time() - start_time)

    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_history.png")
    plt.show()

    print 'Test'
    print '=-=-='
    #calcualate accuracy on test dataset
    predictions, _ = model.predict(model.test_data, './weights/%s'%model.config.model_name) #make prediction by using learned model(including recover saved parameters)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()
    print 'Test acc: {}'.format(test_acc)

if __name__ == "__main__":
        test_RNN()
