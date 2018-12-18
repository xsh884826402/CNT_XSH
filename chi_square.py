import os
from collections import Counter


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(list, count):
    word_counts = Counter(list)
    lists = sorted(dict(word_counts).items(), key=lambda d: d[1], reverse=True)
    dic = {lists[i][0]: lists[i][1] for i in range(len(lists))}
    dic.pop('')
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {wor: inde for inde, wor in enumerate(vocabulary_inv)}
    return vocabulary, dic


def chi_square(dic1, num1, dic2, num2):
    # dic1 总词典
    # num1 总文本数
    # dic2 分类词典
    # num2 分类文本数
    # dic 分类词卡方结果
    dic = {}
    for word, n2 in dic2.items():
        a = n2
        b = dic1[word] - n2
        c = num2 - a
        d = num1 - num2 - b

        result = (a * d - b * c) ** 2 / ((a + c) * (a + b) * (b + d) * (c + d))  # 计算公式
        dic[word] = result
    return dic


def make_list(path):
    list = []
    files = os.listdir(path)
    count = 0
    for file in files:
        count += 1
        with open(path + '/' + file, 'r', encoding='utf8') as f:
            words = f.read().split(' ')
            for word in words:
                if not word == ' ':
                    list.append(word)
    return list, count


def all_chi_square(path):
    # 返回各分类的卡方检验
    dirs = os.listdir(path)
    # 得到所有文件的word_list和文件count
    all_list = []
    all_count = 0
    lists = {}
    counts = {}
    for dir in dirs:
        list, count = make_list(path + '/'+dir)
        for word in list:
            all_list.append(word)
        all_count += count
        lists[dir] = list
        counts[dir] = count
    # 构建分类vab和总vab
    vab_list = {}
    for dir in dirs:
        vab, dict = build_vocab(lists[dir], counts[dir])
        vab_list[dir] = dict
    all_vab = {word: sum(vab[word] for label, vab in vab_list.items() if word in vab) for word in all_list}
    # 建立各分类的卡方list
    all_chisquare = {}
    for dir in dirs:
        chi_dic = chi_square(all_vab, all_count, vab_list[dir], counts[dir])
        all_chisquare[dir] = chi_dic
    all_chisquare_new = {}
    for label, chisquare in all_chisquare.items():
        temp_xsh = sorted(chisquare.items(), key=lambda d: d[1], reverse=True)
        all_chisquare_new[label] = temp_xsh[:int(0.8 * len(temp_xsh))]
        print(all_chisquare_new[label])
    # dic_temp = sorted(chisquare.items() for label, chisquare in all_chisquare.items(),key = lambda d:d[1],reverse=True)
    # all_chisquare = {label:sorted(chisquare.items(), key = lambda d:d[1],reverse=True) for label, chisquare in all_chisquare.items()}
    return all_chisquare_new


def delete_all_few_words(path, value):
    all_chisquare = all_chi_square(path)  # 求所有词语的卡方检验值
    for label in all_chisquare.keys():
        # all_chisquare[label] = all_chisquare[label][:num_words]
        # with open(path+label+'.txt','w',encoding='utf8') as f: ##这里注释
        # number = 0
        # for data in all_chisquare[label]:
        # number += 1
        # f.writelines(str(number)+' '+str(data[0])+' '+str(data[1])+'\n')
        # list = [pair[0] for pair in all_chisquare[label]]
        list1 = [pair[0] for pair in all_chisquare[label] ]
        all_chisquare[label] = list1
    dirs = os.listdir(path)  # 删除各病历下面的词语
    for dir in dirs:
        if not dir.endswith('.txt'):
            files = os.listdir(path + '/'+dir)
            for file in files:
                rlist = ''
                with open(path +'/'+ dir + '/' + file, 'r', encoding='utf8') as f:
                    s = f.read()
                    print('old', s)
                    words = s.split(' ')
                    for word in words:
                        if word in all_chisquare[dir] and word != '':
                            rlist += word + ' '
                    print('new ', rlist)
                with open(path +'/'+ dir + '/' + file, 'w', encoding='utf8') as f:
                    f.write(rlist)

                # path2 = 'data/test/'
                # dirs2 = os.listdir(path2)
                # for dir in dirs2:
                # 	if not dir.endswith('.txt'):
                # 		files = os.listdir(path2+dir)
                # 		for file in files :
                # 			rlist = ''
                # 			with open(path2+dir+'/'+file,'r',encoding='utf8') as f:
                # 				s = f.read()
                # 				#print(s)
                # 				words = s.split(' ')
                # 				for word in words :
                # 					if word in all_chisquare[dir] and word != '':
                # 						rlist += word + ' '
                # 				#print(rlist)
                # 			with open(path2+dir+'/'+file,'w',encoding='utf8') as f:
                # 				f.write(rlist)


def delete_words(value=1):
    if not value == 1:
        path = 'data/train/'
        delete_all_few_words(path, value)


if __name__ == '__main__':
    # train_and_test_no_label.split_train()
    path = 'data/data15_2/train'
    value = 0.005
    delete_all_few_words(path, value)