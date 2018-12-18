#encoding:utf-8
from model_cnn_wordvec import *
import os
import pickle
from data_build import *
from keras.preprocessing import sequence
from keras.utils import np_utils
from cnews_loader import *
import numpy as np
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
        loss, acc =sess.run([model.loss, model.acc],feed_dict =feed_dict)
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
    print("config Tensorboard and saver")

    tensorboard_dir = './tensorboard'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy",model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    #配置Saver
    saver = tf.train.Saver()
    save_dir = './checkpoints/textcnn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("loading data")
    #输入数据
    file_cidian = open('./cidian_1.pkl', 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)
    print('run',indexword['无痛经'])
    #这里控制输入数据
    x_train, y_train = data_build('./data/病历数据82分均衡/train')
    x_test, y_test = data_build('./data/病历数据82分均衡/test')
    # 转换成index
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    #填充序列
    max_length = 100
    x_train = sequence.pad_sequences(x_train, max_length)
    x_test = sequence.pad_sequences(x_test, max_length)

    y_train = np_utils.to_categorical(y_train, 40)
    y_test = np_utils.to_categorical(y_test, 40)

    #创建session
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Training and evaluate")
    total_batch = 0
    best_acc_val = 0.0
    last_improved =0
    require_improvement = 1000
    flag = False

    #2018-3-22修改，为了输出embedding
    feed_dict_eval = {
        model.input_x: x_train,
        model.input_y: y_train,
        model.keep_prob: config.dropout_keep_prob
    }
    print('embedding', np.array(model.embedding[1, :]))
    print('embedding_inputs_1', model.embedding_inputs_1[0, 1, :].eval(feed_dict=feed_dict_eval))
    for epoch in range(config.num_epochs):
        print('Epoch',epoch)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: config.dropout_keep_prob
            }
            if total_batch % config.save_per_batch:
                s = session.run(merged_summary, feed_dict =feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc],feed_dict= feed_dict)
                #这里有疑问,这里要补充
                loss_val, acc_val = evaluate(session, x_test, y_test)

                if acc_val >best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess = session,save_path=save_dir)
                    improved_str = '*'
                else:
                    improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, '
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))
            session.run(model.optim, feed_dict = feed_dict)
            total_batch += 1
        print('embedding_inputs', np.array(model.embedding_inputs))


def test():
    print("loading test data")
    file_cidian = open('./cidian_1.pkl', 'rb', )
    indexword = pickle.load(file_cidian)
    vectorword = pickle.load(file_cidian)
    x_train, y_train = data_build('./data/data15/train')
    x_test, y_test = data_build('./data/data15/test')
    # 转换成index
    x_train = word_to_index(x_train, indexword)
    x_test = word_to_index(x_test, indexword)
    # print('x_train', x_train)

    # 填充序列
    max_length = 100
    x_train = sequence.pad_sequences(x_train, max_length)
    x_test = sequence.pad_sequences(x_test, max_length)

    y_train = np_utils.to_categorical(y_train, 40)
    y_test = np_utils.to_categorical(y_test, 40)



if __name__ =='__main__':
    print("config CNN")
    config = Config()
    model = TextCnn(config)
    train()
