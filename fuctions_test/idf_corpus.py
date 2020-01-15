# author:fighting
import math
import xlrd
import jieba
corpus_path = '/Users/fighting/PycharmProjects/BiLSTM_MClassification/data/intention/Task1data/corpus_list.xlsx'

# 将sentences分词
def getWords(sentences):
    words = []
    new_words = open('/Users/fighting/PycharmProjects/BiLSTM_MClassification/data/intention/specialWords', encoding='utf-8').read().split('\n')
    for word in new_words:
        jieba.suggest_freq(word, True)
    words_cut = jieba.cut(sentences, cut_all=False)
    for word in words_cut:
        words.append(word)
    return words

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

all_datas, all_labels = getCorpus(corpus_path)

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

def get_classwords_set():
    class_sentence = getClass_sentences()
    class_words = []
    for key in class_sentence.keys():
        words = getWords(class_sentence[key])
        class_words.append(words)
    return class_words

def get_idf_dic(class_words):
    idf_dic = {}
    doc_count = len(class_words)
    for i in range(doc_count):
        new_content = class_words[i]
        for word in set(new_content):
            if len(word) > 1:
                idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    for k, v in idf_dic.items():
        w = k
        p = '%.10f' % (math.log(doc_count / (1.0 + v)))  # 结合上面的tf-idf算法公式
        if w > u'\u4e00' and w <= u'\u9fa5':  # 判断key值全是中文
            idf_dic[w] = p

    with open("/Users/fighting/PycharmProjects/BiLSTM_MClassification/data/intention/wdic.txt", 'w', encoding='utf-8') as f:
        for k in idf_dic:
            if k !='\n':
                f.write(k + ' ' + str(idf_dic[k]) + '\n')  # 写入txt文件，注意utf-8，否则jieba不认






class_words = get_classwords_set()
print(class_words)
get_idf_dic(class_words)