from run_nnlm import *
import tensorflow as tf
import pickle
save_dir1 = './checkpoints/zangfu/'
save_dir2 = './checkpoints/BaGang/'
save_dir3 = './checkpoints/QiXueJinYe/'
save_dir4 = './checkpoints/WeiQiYingXue/'
save_dir5 = './checkpoints/SanJiao/'

# def predict():
#     # print('Hello')
#训练
# load_fenci.py 加载
config = Config(10,7)#num_epoch, num_classes
model1 = TextCnn(config)
# train1(model1, save_dir1)
y_str1,y_return =test1(model1, save_dir1)
# print(y_return, np.shape(y_return))
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'wb') as f:
    pickle.dump(y_return, f)
    # print('Store Label Success')
tf.reset_default_graph()

config = Config(20,8)
model2 = TextCnn(config)
# train2(model2, save_dir2)
y_str2 = test2(model2, save_dir2)

tf.reset_default_graph()

config = Config(20,7)
model3 = TextCnn(config)
# train3(model3, save_dir3)
y_str3 = test3(model3, save_dir3)

tf.reset_default_graph()
config = Config(20, 3)
model4 = TextCnn(config)
# train4(model4, save_dir4)
y_str4 = test4(model4, save_dir4)

tf.reset_default_graph()
config = Config(20, 3)
model5 = TextCnn(config)
# train5(model5, save_dir5)
y_str5 = test5(model5, save_dir5)
#开始对病历预测结果

str_dir ='./y_str1'
#按照脏腑 气血津液，八纲，卫气营血，三焦
with open(str_dir, 'wb') as f:
    pickle.dump(y_str1, f)
    pickle.dump(y_str3, f)
    pickle.dump(y_str2, f)
    pickle.dump(y_str4, f)
    pickle.dump(y_str5, f)
    print('Store Success')

