import tensorflow as tf
import numpy as np
class Config( ):
    embedding_dim = 64
    seq_length = 100
    num_classes = 40
    vocab_size = 5000

    num_layer = 3
    hidden_dim = 128
    type = 'gru'

    drop_keep_prob = 1
    learning_rate = 1e-3
    batch_size = 64
    n_epoch = 50

    print_per_batch = 30
    save_per_batch = 10
class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x =tf.placeholder(tf.int32, (None, self.config.seq_length), name='input_x')
        self.input_y = tf.placeholder(tf.float32, (None, self.config.num_classes), name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        def Gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)
        def dropout():
            if self.config.type =='lstm':
                cell = lstm_cell()
            else:
                cell = Gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.config.drop_keep_prob)
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            cells = [dropout() for _ in range(self.config.num_layer)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple = True)

            _outputs, _ =tf.nn.dynamic_rnn(cell = rnn_cell, inputs = embedding_input, dtype=tf.float32)
            shape = tf.shape(_outputs)
            print('output_shape',_outputs.shape)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            print('output_shape', last.shape)
        with tf.name_scope('score'):
            fc = tf.layers.dense(last, self.config.hidden_dim,name = 'fcl')
            print('fc', fc.shape)
            fc = tf.layers.dropout(fc,self.keep_prob)
            fc =tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name = 'fc2')
            self.y_pred_cls =tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss_accuracy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,labels=self.input_y)
            self.loss =tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate= self.config.learning_rate).minimize(self.loss)

            correct_prde = tf.equal(self.y_pred_cls,tf.arg_max(self.input_y,1))
            self.acc = tf.reduce_mean(tf.cast(correct_prde,tf.float32))

