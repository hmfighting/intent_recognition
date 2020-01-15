# author:fighting
# 使用jieba对语料进行词汇分词
# 使用word2vec对词汇进行向量训练
import jieba
import jieba.analyse
import os
from gensim.models.word2vec import Word2Vec
import xlrd

new_words = open('../data/intention/specialWords', encoding='utf-8').read().split('\n')
for word in new_words:
    jieba.suggest_freq(word, True)

file_path = './data/LCQMC'
def getWords(sentences):
    train_words = []
    words = []
    for string in sentences:
        temp = []
        temp_char = []
        words_cut = jieba.cut(string, cut_all=False)
        for word in words_cut:
            temp.append(word)
            words.append(word)
        train_words.append(temp)
        chars = list(string)
        for ch in chars:
            temp_char.append(ch)
        train_words.append(temp_char)
    return words, train_words

# 获取Task1data中的数据
def get_excel_data(filepath):
    # 获取数据
    data = xlrd.open_workbook(filepath)
    # 获取sheet
    table = data.sheet_by_name('Sheet')  # 注意读取的表中sheet必须重新命名为question
    # 获取总行数
    nrows = table.nrows
    # 获取总列数
    nclos = table.ncols
    ques_list = []
    for i in range(1, nrows):
        question = str(table.cell(i, 0).value).strip()
        if len(question.strip()) >= 1:
            ques_list.append(question)
    return ques_list

def get_sentences():
    sentences = []
    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), 'r') as lcqmc:
            for line in lcqmc:
                linedict = eval(line)
                word = linedict['sentence1']
                poss = linedict['sentence2']
                sentences.append(word)
                sentences.append(poss)
    # texts = open('../gensim_word2vec/data/data_text_noRepeat').read().split('\n')
    corpus_texts = get_excel_data('../data/intention/Task1data/corpus_list.xlsx')
    sentences = sentences + corpus_texts
    # sentences = corpus_texts
    return sentences


sentences = get_sentences()
for i in range(5):
    print(sentences[i])
print("data_text size:", len(sentences))
words, train_words = getWords(sentences)


model = Word2Vec(train_words, size=128, window=4, min_count=1, sg=1, workers=2)
model.init_sims(replace=True)
model.save('./wordVec_model/word2vecModel_public')
print(model['卫视'])
# print(pro = model.n_similarity("央", "卫视"))