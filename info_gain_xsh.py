import os
import numpy as np
def calu(x):
    return 0 if x == 0 else x * np.log2(x)
def info_gain(path):
    count_lei_file={}
    count_sum_file=0
    word_list = []
    # count the amount of files each class
    for lei in os.listdir(path):
        count_lei_file[lei]=len(os.listdir(path+'/'+lei))
        count_sum_file +=len(os.listdir(path+'/'+lei))
        for file in os.listdir(path+'/'+lei):
            with open(path+'/'+lei+'/'+file, 'r', encoding='utf-8') as f:
                words =f.read().split(' ')
                for word in words:
                    if word !='':
                        word_list.append(word)
    word_list=list(set(word_list))
    #print(count_sum_file)
    ig_0 = 0.0
    for lei, count in count_lei_file.items():
        ig_0 -= calu(count*1.0/count_sum_file)
    word_ig={}# word_info_gain_dict
    #computing word info_gain for each word
    for word in word_list:
        count_lei_word_showinfile = {}
        word_show_num = 0
        a = 0
        b = 0
        for lei in os.listdir(path):
            count_lei_word =0
            for file in os.listdir(path+'/'+lei):
                with open(path+'/'+lei+'/'+file, 'r', encoding='utf-8') as f:
                    words = f.read().split(' ')
                    if word in words:
                        count_lei_word += 1
                        word_show_num += 1
            count_lei_word_showinfile[lei] = count_lei_word

        for lei in os.listdir(path):
            a += calu(count_lei_word_showinfile[lei] * 1.0 / word_show_num)
            b += calu((count_lei_file[lei] - count_lei_word_showinfile[lei]) * 1.0 / (count_sum_file - word_show_num))
        pt =word_show_num*1.0/count_sum_file
        pt_1 = 1-pt
        a=pt*a
        b=pt_1*b
        ig_1 =ig_0+a+b
        word_ig[word] = ig_1
        # if word=='烘热汗出':
        #     print(word,ig_1)
        #     print('cishu',word_show_num)
        #     print('all', count_sum_file)

    #computing job is done
    word_info_gain = sorted(word_ig.items(), key=lambda d:d[1],reverse=True )
    return word_info_gain
def delete_word(path, info_gain, count):
    length =len(info_gain)
    i=0
    word_list=[]
    for item in info_gain:
        if i < int(count*length):
            word_list.append(item[0])
            i+= 1
    for lei in os.listdir(path):
        for file in os.listdir(os.path.join(path, lei)):
            with open(os.path.join(path,lei,file), 'r', encoding= 'utf-8') as fopen:
                old_count = 0
                new_count = 0
                line = fopen.read().split(' ')
                new_line =[]
                for word in line:
                    old_count +=1
                    if word in word_list:
                        new_line.append(word)
                        new_count += 1
                print('old',old_count,'    new', new_count)#compare how many words are cutted
            with open(os.path.join(path,lei,file), 'w', encoding ='utf-8') as f:
                for word in new_line:
                    #print(word)
                    f.write(word +' ')
                #print('line')
def add_dir(path, path1, path2):# merge two diretory
    os.makedirs(path, exist_ok=True)
    for lei in os.listdir(path1):
        os.makedirs(os.path.join(path, lei), exist_ok=True)
    for lei in os.listdir(path1):
        for file in os.listdir((os.path.join(path1,lei))):
            with  open(os.path.join(path,lei, file), 'w', encoding ='utf-8') as f:
                f.write(open(os.path.join(path1, lei, file), 'r', encoding='utf-8').read())
    for lei in os.listdir(path2):
        for file in os.listdir((os.path.join(path2,lei))):
            with  open(os.path.join(path,lei, file), 'w', encoding ='utf-8') as f:
                f.write(open(os.path.join(path2, lei, file), 'r', encoding='utf-8').read())



if __name__ =='__main__':
    dir = 'data/data15_1'
    # add_dir(dir + '/add', dir + '/train', dir + '/test')

    info_g = info_gain(dir + '/train')
    delete_word(dir + '/train', info_g, 0.8)
    #delete_word(dir + '/test', info_g, 0.8)