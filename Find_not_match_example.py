from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from model_nnlm import *
import Write_xls
import pickle
import data_build


#准备训练数据x和y
with open('./data/label_transfer_dict.pkl', 'rb') as f:
    dict = pickle.load(f)
print('ceshi', dict['肺经蕴热'])

# y_keras = np_utils.to_categorical(y,num_classes=40)
# print('keras',y_keras)
# print('标签个数',le.classes_,)
# print('标准化',le.transform(["肺经蕴热"]))
# print(y)

clf = MLPClassifier()
x = []
mlb1 = preprocessing.MultiLabelBinarizer()
mlb2 = preprocessing.MultiLabelBinarizer()
mlb3 = preprocessing.MultiLabelBinarizer()
mlb4 = preprocessing.MultiLabelBinarizer()
mlb5 = preprocessing.MultiLabelBinarizer()
mlb1.fit(['心肝脾肺肾胆胃'])
mlb2.fit(['气','血','湿','痰','泛','水','瘀'])
mlb3.fit(['阴阳表里虚实寒热'])
mlb4.fit(['卫','气','血'])
mlb5.fit(['上','中','下'])
def x_to_vector(x):
    #x 是一个一维的列表 列表中分为五个元素
    label1 = x[0]
    if label1 != label1:
        label1 = ''
    x_temp = mlb1.transform([label1])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = x_temp

    label2 = x[1]
    if label2 != label2:
        label2 = ''
    x_temp = mlb2.transform([label2])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label3 = x[2]
    if label3 != label3:
        label3 = ''
    x_temp = mlb3.transform([label3])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label4 = x[3]
    if label4 != label4:
        label4 = ''
    x_temp = mlb4.transform([label4])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    label5 = x[4]
    if label5 != label5:
        label5 = ''
    x_temp = mlb5.transform([label5])
    x_temp = np.reshape(x_temp, x_temp.size)
    x1 = np.append(x1, x_temp)

    return x1


#将辩证分词保存成词典
dict_qingxi ={}
for k in dict.keys():
    x_train_temp = []
    for j in range(3, 8):
        if dict[k][j] !=dict[k][j]:
            x_train_temp.append('')
        else:
            x_train_temp.append(dict[k][j])
    dict_qingxi[k] = x_train_temp


str_dir = './y_str1'
with open(str_dir, 'rb') as f:
    y_str1 = pickle.load(f)
    y_str2 = pickle.load(f)
    y_str3 = pickle.load(f)
    y_str4 = pickle.load(f)
    y_str5 = pickle.load(f)

y_bianzheng = []
for i in range(len(y_str1)):
    y_bianzheng_temp = []
    y_bianzheng_temp.append(y_str1[i])
    y_bianzheng_temp.append(y_str2[i])
    y_bianzheng_temp.append(y_str3[i])
    y_bianzheng_temp.append(y_str4[i])
    y_bianzheng_temp.append(y_str5[i])
    y_bianzheng.append(y_bianzheng_temp)
# print('bianzheng',y_bianzheng)
save_Test_label_dir = './Test_Label'
with open(save_Test_label_dir, 'rb') as f:
    y_label = pickle.load(f)
text,labels = data_build.data_build_label('./data/bingli_exp_result/test')
Not_match_list = []
Not_match_text = []
Not_match_label = []
for i in range(len(y_label)):
    Not_match_list_temp = []
    Leibie = y_label[i]
    for j in range(5):
        str_temp = set(y_bianzheng[i][j])
        if(str_temp !=set(dict_qingxi[Leibie][j])):
            Not_match_list_temp.append(Leibie)
            Not_match_list_temp.append(y_bianzheng[i])
            Not_match_list_temp.append(dict_qingxi[Leibie])
            Not_match_list.append(Not_match_list_temp)
            Not_match_text.append(text[i])
            Not_match_label.append(labels[i])
            break
Write_xls.list_to_xls4(Not_match_list,Not_match_text,Not_match_label,"不匹配结果_1.xls")






