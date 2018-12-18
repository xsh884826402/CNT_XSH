from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import numpy as np
from data_build import *
def get_multilabel_ZangFu(y_train):
    labels = ['心肝脾肺肾胆胃']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][3]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][3])
    print(y_temp)
    print(mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    print('type',type(y_multilabel))
    print('shape', np.shape(y_multilabel))
    print(mlb.inverse_transform(y_multilabel))
    return y_multilabel
def get_inverse_multilabel_ZangFu(y):
    labels = ['心肝脾肺肾胆胃']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_BaGang(y_train):
    labels = ['阴阳表里虚实寒热']
    index = 5
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_BaGang(y):
    labels = ['阴阳表里虚实寒热']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_QiXueJinYe(y_train):
    labels = ['气','血','湿','痰','泛','水','瘀']
    index = 4
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_QiXueJinye(y):
    labels = ['气', '血', '湿', '痰', '泛', '水', '瘀']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_WeiQiYingXue(y_train):
    labels = ['卫','气','血']
    index = 6
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_WeiQiYingXue(y):
    labels = ['卫', '气', '血']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
def get_multilabel_SanJiao(y_train):
    labels = ['上','中','下']
    index = 7
    #index 表明在哪个角度去看辩证
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    with open('./data/label_transfer_dict.pkl', 'rb') as f:
        label_transfer_dict = pickle.load(f)

    y_temp = []
    count = 0
    for item in y_train:
            # print(item)
            temp = label_transfer_dict[item][index]
            if temp != temp:
                y_temp.append('')
            else:
                y_temp.append(label_transfer_dict[item][index])
    print(y_temp)
    print('in_get_multilabel',mlb.classes_)
    y_multilabel = mlb.transform(y_temp)
    print(y_multilabel)
    return y_multilabel
def get_inverse_multilabel_SanJiao(y):
    labels = ['上', '中', '下']
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    y_label = mlb.inverse_transform(y)
    return y_label
y = np.array([0, 0.,0.,1,0.,0,0])
print(np.reshape(y,(1,7)))
# get_inverse_multilabel_ZangFu(y)