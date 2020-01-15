import numpy as np
import pickle
import xlrd
import jieba
import jieba.analyse
from os.path import exists, join
import math
from gensim.models.word2vec import Word2Vec
# word vetor representation
model = Word2Vec.load('fuctions_test/wordVec_model/word2vecModel_public')
jieba.analyse.set_idf_path(r'./data/intention/wdic.txt')
# get the questions and intentions from courpus
def getCorpus(file_path):
    # get the text from file_path
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name('Sheet')  #notice: read the sheet named "Sheet"
    # all rows
    nrows = table.nrows
    # all cols
    nclos = table.ncols
    ques_list = []
    intent_list = []
    for i in range(1, nrows):
        question = str(table.cell(i, 0).value).strip()
        intention = str(table.cell(i, 1).value).strip()
        # temp: save the labels of each question，the format of saving：[question, label of category]
        if len(question) >= 1:
            if question in ques_list:
                continue
            ques_list.append(question)
            intention = int(float(intention))
            intent_list.append(intention)
    return ques_list, intent_list

train_path = 'data/intention/Task1data/Labeled_ques_list.xlsx'
dev_path = 'data/intention/Task1data/dev_list.xlsx'
test_path = 'data/intention/Task1data/test_list.xlsx'
corpus_path = 'data/intention/Task1data/corpus_list.xlsx'
# train dataset
train, train_label = getCorpus(train_path)
print("train_data_size:", len(train), "train_label_size:", len(train_label))
# validation dataset
dev, dev_label = getCorpus(dev_path)
print("dev_data_size:", len(dev), "dev_label_size:", len(dev_label))
# test dataset
test, test_label = getCorpus(test_path)
# corpus: we can get the keywords of each category from the corpus
print("test_data_size:", len(test), "test_label_size:", len(test_label))
all_datas, all_labels = getCorpus(corpus_path)
print('测试数据集中测试样本的类别标签：', test_label)

# get the sentences of each category from the all_datas and all_labels
def getClass_sentences():
    class_sentences = {}
    if len(all_datas) == len(all_labels):
        for i in range(len(all_labels)):
            if float(all_labels[i]) == 1:
                if '1' in class_sentences.keys():
                    class_sentences['1'] = class_sentences['1'] + all_datas[i]
                else:
                    class_sentences['1'] = all_datas[i]
            if float(all_labels[i]) == 2:
                if '2' in class_sentences.keys():
                    class_sentences['2'] = class_sentences['2'] + all_datas[i]
                else:
                    class_sentences['2'] = all_datas[i]
            if float(all_labels[i]) == 3:
                if '3' in class_sentences.keys():
                    class_sentences['3'] = class_sentences['3'] + all_datas[i]
                else:
                    class_sentences['3'] = all_datas[i]
            if float(all_labels[i]) == 4:
                if '4' in class_sentences.keys():
                    class_sentences['4'] = class_sentences['4'] + all_datas[i]
                else:
                    class_sentences['4'] = all_datas[i]
            if float(all_labels[i]) == 5:
                if '5' in class_sentences.keys():
                    class_sentences['5'] = class_sentences['5'] + all_datas[i]
                else:
                    class_sentences['5'] = all_datas[i]
            if float(all_labels[i]) == 6:
                if '6' in class_sentences.keys():
                    class_sentences['6'] = class_sentences['6'] + all_datas[i]
                else:
                    class_sentences['6'] = all_datas[i]
            if float(all_labels[i]) == 7:
                if '7' in class_sentences.keys():
                    class_sentences['7'] = class_sentences['7'] + all_datas[i]
                else:
                    class_sentences['7'] = all_datas[i]
            if float(all_labels[i]) == 8:
                if '8' in class_sentences.keys():
                    class_sentences['8'] = class_sentences['8'] + all_datas[i]
                else:
                    class_sentences['8'] = all_datas[i]
            if float(all_labels[i]) == 9:
                if '9' in class_sentences.keys():
                    class_sentences['9'] = class_sentences['9'] + all_datas[i]
                else:
                    class_sentences['9'] = all_datas[i]
            if float(all_labels[i]) == 10:
                if '10' in class_sentences.keys():
                    class_sentences['10'] = class_sentences['10'] + all_datas[i]
                else:
                    class_sentences['10'] = all_datas[i]
            if float(all_labels[i]) == 11:
                if '11' in class_sentences.keys():
                    class_sentences['11'] = class_sentences['11'] + all_datas[i]
                else:
                    class_sentences['11'] = all_datas[i]
            if float(all_labels[i]) == 12:
                if '12' in class_sentences.keys():
                    class_sentences['12'] = class_sentences['12'] + all_datas[i]
                else:
                    class_sentences['12'] = all_datas[i]
            if float(all_labels[i]) == 13:
                if '13' in class_sentences.keys():
                    class_sentences['13'] = class_sentences['13'] + all_datas[i]
                else:
                    class_sentences['13'] = all_datas[i]
            if float(all_labels[i]) == 14:
                if '14' in class_sentences.keys():
                    class_sentences['14'] = class_sentences['14'] + all_datas[i]
                else:
                    class_sentences['14'] = all_datas[i]
            if float(all_labels[i]) == 15:
                if '15' in class_sentences.keys():
                    class_sentences['15'] = class_sentences['15'] + all_datas[i]
                else:
                    class_sentences['15'] = all_datas[i]
            if float(all_labels[i]) == 16:
                if '16' in class_sentences.keys():
                    class_sentences['16'] = class_sentences['16'] + all_datas[i]
                else:
                    class_sentences['16'] = all_datas[i]
            if float(all_labels[i]) == 17:
                if '17' in class_sentences.keys():
                    class_sentences['17'] = class_sentences['17'] + all_datas[i]
                else:
                    class_sentences['17'] = all_datas[i]
            if float(all_labels[i]) == 18:
                if '18' in class_sentences.keys():
                    class_sentences['18'] = class_sentences['18'] + all_datas[i]
                else:
                    class_sentences['18'] = all_datas[i]
            if float(all_labels[i]) == 19:
                if '19' in class_sentences.keys():
                    class_sentences['19'] = class_sentences['19'] + all_datas[i]
                else:
                    class_sentences['19'] = all_datas[i]
            if float(all_labels[i]) == 20:
                if '20' in class_sentences.keys():
                    class_sentences['20'] = class_sentences['20'] + all_datas[i]
                else:
                    class_sentences['20'] = all_datas[i]
            if float(all_labels[i]) == 21:
                if '21' in class_sentences.keys():
                    class_sentences['21'] = class_sentences['21'] + all_datas[i]
                else:
                    class_sentences['21'] = all_datas[i]
            if float(all_labels[i]) == 22:
                if '22' in class_sentences.keys():
                    class_sentences['22'] = class_sentences['22'] + all_datas[i]
                else:
                    class_sentences['22'] = all_datas[i]
            if float(all_labels[i]) == 23:
                if '23' in class_sentences.keys():
                    class_sentences['23'] = class_sentences['23'] + all_datas[i]
                else:
                    class_sentences['23'] = all_datas[i]
            if float(all_labels[i]) == 24:
                if '24' in class_sentences.keys():
                    class_sentences['24'] = class_sentences['24'] + all_datas[i]
                else:
                    class_sentences['24'] = all_datas[i]
            if float(all_labels[i]) == 25:
                if '25' in class_sentences.keys():
                    class_sentences['25'] = class_sentences['25'] + all_datas[i]
                else:
                    class_sentences['25'] = all_datas[i]
            if float(all_labels[i]) == 26:
                if '26' in class_sentences.keys():
                    class_sentences['26'] = class_sentences['26'] + all_datas[i]
                else:
                    class_sentences['26'] = all_datas[i]
            if float(all_labels[i]) == 27:
                if '27' in class_sentences.keys():
                    class_sentences['27'] = class_sentences['27'] + all_datas[i]
                else:
                    class_sentences['27'] = all_datas[i]
            if float(all_labels[i]) == 28:
                if '28' in class_sentences.keys():
                    class_sentences['28'] = class_sentences['28'] + all_datas[i]
                else:
                    class_sentences['28'] = all_datas[i]
            if float(all_labels[i]) == 29:
                if '29' in class_sentences.keys():
                    class_sentences['29'] = class_sentences['29'] + all_datas[i]
                else:
                    class_sentences['29'] = all_datas[i]
            if float(all_labels[i]) == 30:
                if '30' in class_sentences.keys():
                    class_sentences['30'] = class_sentences['30'] + all_datas[i]
                else:
                    class_sentences['30'] = all_datas[i]
            if float(all_labels[i]) == 31:
                if '31' in class_sentences.keys():
                    class_sentences['31'] = class_sentences['31'] + all_datas[i]
                else:
                    class_sentences['31'] = all_datas[i]
    else:
        print('读取labeled_ques.xlsx数据有误')
    return class_sentences


# 将sentences分词
def getWords(sentences):
    words = []
    words_num = [] # record the word number included each sentence
    new_words = open('data/intention/specialWords', encoding='utf-8').read().split('\n')
    for word in new_words:
        jieba.suggest_freq(word, True)
    for string in sentences:
        words_cut = jieba.cut(string, cut_all=False)
        temp = []   # Temporarily store the words contained in each sentence
        temp_char = []
        for word in words_cut:
            temp.append(word)
        words_num.append(len(temp))
        words.append(temp)
    return words, words_num

train_words, train_nums = getWords(train)
dev_words, dev_nums = getWords(dev)
test_words, test_nums = getWords(test)
print('train_words:', train_words)
print('train训练语料中句子的长度：', train_nums)


# acquire the keyword of category according to the sentence set of each category
def get_class_keyword(class_sentence):
    words_weight = {}
    keywords = jieba.analyse.extract_tags(class_sentence, topK=10, withWeight=True)  #set the number of keyword
    for item in keywords:
        words_weight[item[0]] = item[1]
    return words_weight

# 获取7个类别的关键词
# class_keyword中key为类别序号，value是一个字典类型，里面存放了5个关键词以及关键词的概率
# '1.0': '查件', '2.0': '收派件', '3.0': '业务问答', '4.0': '生活问答', '5.0': '寄件', '6.0': '抱怨', '7.0': '问候'
def get_seven_keyword():
    class_keyword = {}
    class_sentences = getClass_sentences()  #class_sentences中存放了每个类别下的句子集合
    for key in class_sentences:
        words_weight = get_class_keyword(class_sentences[key])
        class_keyword[key] = words_weight
    return class_keyword

class_keyword = get_seven_keyword()
print("每种类别关键词：", class_keyword)

# Calculate the distance between two words
def get_words_similarity(words1, words2):
    pro = model.wv.n_similarity(words1, words2)
    return pro

# acquire the vector of Chinese word
def get_word_vector(word):
    return model[word]


# Gets the most similar weights for the input words in each category
def getClassify_feature(word):
    class_keyword = get_seven_keyword()
    word_classWeight = {}
    for key in class_keyword:
        max_weight = 0
        values_keywords = class_keyword[key]
        i = 1
        for k in values_keywords:
            # temp = get_words_similarity(word, k)*values_keywords[k]
            temp = get_words_similarity(word, k) + (1/(1+math.exp(i)))
            i += 1
            if max_weight < temp:
                max_weight = temp
        word_classWeight[key] = max_weight
        # print(key + ":" + strs)
    return word_classWeight

# 获取输入文本与每个类别的距离
def getInputFileDistanceWithClassify(data_set):
    distance_vectors = []
    for sentence in data_set:
        sen_vector = []
        n = 0
        if len(sentence) > 25:
            n = 25
        else:
            n = len(sentence)
        for i in range(n):
            word = sentence[i]
            temp = getClassify_feature(word)
            print(word + ":" + str(temp))
            sort_temp = sorted(temp.items(), key=lambda x: x[0])  # 对字典类型的集合进行排序
            distance_vector = []
            for j in range(len(sort_temp)):
                distance_vector.append(sort_temp[j][1])
            distance_vector = np.asarray(distance_vector)
            sen_vector.append(distance_vector)
        for i in range(len(sentence), 25):
            distance_vector= [0 for _ in range(31)]
            distance_vector = np.asarray(distance_vector)
            sen_vector.append(distance_vector)
        sen_vector = np.asarray(sen_vector)
        distance_vectors.append(sen_vector)
    return distance_vectors


test_distancce = getInputFileDistanceWithClassify(test_words)
test_distancce = np.array(test_distancce)
train_distance = getInputFileDistanceWithClassify(train_words)
train_distance = np.array(train_distance)
dev_distance = getInputFileDistanceWithClassify(dev_words)
dev_distance = np.array(dev_distance)
print('dev_distance_size:', dev_distance.shape)   #（31，25，31）



# 获取输入文本的向量
def getInputFileVector(data_set):
    vectors = []
    for sentence in data_set:
        sentence_vector = []
        f = 0
        if len(sentence) > 25:
            f = 25
        else:
            f = len(sentence)
        for i in range(f):
            word_vector = get_word_vector(sentence[i])
            word_vector = np.asarray(word_vector)
            sentence_vector.append(word_vector)
        for i in range(f, 25):
            word_vector = [0 for _ in range(128)]
            word_vector = np.asarray(word_vector)
            sentence_vector.append(word_vector)
        sentence_vector = np.asarray(sentence_vector)
        vectors.append(sentence_vector)
    return vectors

# he vectorized representation of the train texts, dev texts and test texts
train_vectors = getInputFileVector(train_words)
train_vectors = np.asarray(train_vectors)
dev_vectors = getInputFileVector(dev_words)
dev_vectors = np.asarray(dev_vectors)
test_vectors = getInputFileVector(test_words)
test_vectors = np.asarray(test_vectors)

print("test测试语料中矩阵的大小", test_vectors.shape)
print("train_vectors_shape:", train_vectors.shape)
print('test_distance_size:', test_distancce.shape)
print('train_distance_size:', train_distance.shape)  #（2100，25，31）
print("")

word_classWeight = getClassify_feature('聚友网')
for k in word_classWeight:
    print(k + ":" + str(word_classWeight[k]))
file_path = 'data/intention/Task1data/Labeled_ques_list.xlsx'

ques_list, intent_list = getCorpus(file_path)
print("语料总数："+ str(len(ques_list)))




print('Starting pickle to file...')
path1 = 'data/intention/'
with open(join(path1, 'data.pkl'), 'wb') as f:
    pickle.dump(train_label, f)
    pickle.dump(dev_label, f)
    pickle.dump(test_label, f)
    pickle.dump(train_vectors, f)
    pickle.dump(dev_vectors, f)
    pickle.dump(test_vectors, f)
    pickle.dump(train_nums, f)
    pickle.dump(dev_nums, f)
    pickle.dump(test_nums, f)
    pickle.dump(train_distance, f)
    pickle.dump(dev_distance, f)
    pickle.dump(test_distancce, f)
    pickle.dump(test, f)
print('Pickle finished')
