import pickle
import numpy as np
import tensorflow as tf
import numpy as np
class Config():
    embedding_dim = 128
    seq_length = 100
    num_classes = 40
    num_filters = 128
    kernel_size = 3
    vocab_size = 5000

    hidden_dim = 256

    dropout_keep_prob = 0.6
    learning_rate = 1e-2

    batch_size =64
    num_epochs = 100

    print_per_batch =10
    save_per_batch =10

class TextCnn():
    def __init__(self,config):
        self.config = config
        # input data
        self.input_x =tf.placeholder(tf.int32,[None, self.config.seq_length],name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name ='keep_prob')
        self.cnn()
    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
        with tf.name_scope("cnn"):
            print('embedding_input',self.embedding_inputs.shape)
            conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters,self.config.kernel_size,padding='valid',  name ='conv')

            print('conv',conv.shape)
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            print('gmp', gmp.shape)
        with tf.name_scope("rnn"):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_filters)
            rnn_cell_1,_ = tf.nn.dynamic_rnn(rnn_cell, inputs=conv, dtype=tf.float32)
            print('rnn_cell,', tf.shape(rnn_cell_1))
            rnn_cell_1 = rnn_cell_1[:, -1, :]
        with tf.name_scope("score"):
            fc = tf.layers.dense(rnn_cell_1, self.config.hidden_dim, name='fcl')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            print('fc', np.shape(fc))
            # fc_1 = tf.layers.dense(fc, 64, name ='fc_22')
            # fc_1 = tf.contrib.layers.dropout(fc_1,self.keep_prob)
            # fc_1 = tf.nn.relu(fc_1)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits,), 1)
            print('fv', np.shape(self.logits))
            print('y_pred_cls', np.shape(self.y_pred_cls))
        with tf.name_scope("optimize"):

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,labels = self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)


            self.optim =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            # self.optim1 =tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            # self.compute = self.optim1.compute_gradients(self.loss)
            # self.apply = self.optim1.apply_gradients(self.compute)
            # print('compute', self.compute)

            # self.tidu = tf.gradients(self.loss, self.embedding_inputs)

        with tf.name_scope("accuracy"):

            correct_pred = tf.equal(tf.arg_max(self.input_y, 1),self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))