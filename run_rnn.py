from model_rnn import *
from data_build import *
from keras.preprocessing import sequence
from keras.utils import np_utils
from cnews_loader import *
import pickle
import os
def evaluate(sess, x, y):
    #评估在某一数据上的准确率和损失
    data_len = len(x)
    batch_eval = batch_iter(x, y, 128)
    total_loss =0.0
    total_acc =0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {
            model.input_x:x_batch,
            model.input_y:y_batch,
            model.keep_prob:1.0
        }
        loss, acc =sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss*batch_len
        total_acc += acc * batch_len
    return total_loss/data_len,total_acc/data_len
def word_to_index(texts, indexword):
    texts_id = []
    for text in texts:
        texts_id_1 = []
        for word in text:
            try:
                texts_id_1.append(indexword[word])
            except:
                texts_id_1.append(0)
        texts_id.append(texts_id_1)
    return texts_id

def train():
    print("Configing Tensorboard and Saver")
    tensorboard_dir = './tensorboard'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir, mode='w')
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accu', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # config saver
    saver_dir = './checkpoints/textrnn'
    saver =tf.train.Saver()
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)

    print("loading trainingdata and Valitiondata")
    x_train, y_train = data_build('./data/data15_2/train')
    x_test, y_test = data_build('./data/data15/test')

    #converse word to index
    file = open('cidian_data15.pkl', 'rb',)
    index = pickle.load(file)
    vector = pickle.load(file)
    x_train = word_to_index(x_train, index)
    x_test = word_to_index(x_test, index)

    y_train =np_utils.to_categorical(y_train, 40)
    y_test = np_utils.to_categorical(y_test, 40)

    #padding the sequence
    max_length = 100
    x_train = sequence.pad_sequences(x_train, max_length)
    x_test = sequence.pad_sequences(x_test, max_length)

    #create session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    print("Training data")
    #begin training
    total_batch =0
    for epoch in range(config.n_epoch):
        print('epoch', epoch +1)
        batch_train = batch_iter(x_train, y_train,config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob : config.drop_keep_prob
            }

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = sess.run([model.loss, model.acc],feed_dict= feed_dict)
                loss_test, acc_test = evaluate(sess, x_test, y_test)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                  + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}'
                print(msg.format(total_batch, loss_train, acc_train, loss_test, acc_test))


            if total_batch % config.save_per_batch == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            sess.run(model.optim, feed_dict = feed_dict)
            total_batch += 1





if __name__=='__main__':
    print("Configuring Rnn")
    config = Config()
    model =TextRNN(config)
    train()