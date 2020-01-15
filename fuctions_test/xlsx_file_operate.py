# author:fighting
import xlrd

def getCorpus(file_path):
    # 获取数据
    data = xlrd.open_workbook(file_path)
    # 获取sheet
    table = data.sheet_by_name('question')
    # 获取总行数
    nrows = table.nrows
    # 获取总列数
    nclos = table.ncols
    ques_list = []
    intent_list = []
    for i in range(1, nrows):
        question = str(table.cell(i, 0).value).strip()
        intention = str(table.cell(i, 1).value).strip()
        if len(question) >= 1 and len(intention) >= 1:
            ques_list.append(question)
            intent_list.append(intention)
    return ques_list, intent_list

file_path = '../data/intention/labeled_ques.xlsx'
ques_list, intent_list =  getCorpus(file_path)
print('ques_list:', ques_list)
print('intent_list:', intent_list)


# 获取每个类别下的句子
def getClass_sentences():
    class_sentences = {}
    ques_list, intent_list = getCorpus(file_path)
    if len(ques_list) == len(intent_list):
        for i in range(len(intent_list)):
            if float(intent_list[i]) == 1:
                if '1' in class_sentences.keys():
                    class_sentences['1'] = class_sentences['1'] + ques_list[i]
                else:
                    class_sentences['1'] = ques_list[i]
            if float(intent_list[i]) == 2:
                if '2' in class_sentences.keys():
                    class_sentences['2'] = class_sentences['2'] + ques_list[i]
                else:
                    class_sentences['2'] = ques_list[i]
            if float(intent_list[i]) == 3:
                if '3' in class_sentences.keys():
                    class_sentences['3'] = class_sentences['3'] + ques_list[i]
                else:
                    class_sentences['3'] = ques_list[i]
            if float(intent_list[i]) == 4:
                if '4' in class_sentences.keys():
                    class_sentences['4'] = class_sentences['4'] + ques_list[i]
                else:
                    class_sentences['4'] = ques_list[i]
            if float(intent_list[i]) == 5:
                if '5' in class_sentences.keys():
                    class_sentences['5'] = class_sentences['5'] + ques_list[i]
                else:
                    class_sentences['5'] = ques_list[i]
            if float(intent_list[i]) == 6:
                if '6' in class_sentences.keys():
                    class_sentences['6'] = class_sentences['6'] + ques_list[i]
                else:
                    class_sentences['6'] = ques_list[i]
            if float(intent_list[i]) == 7:
                if '7' in class_sentences.keys():
                    class_sentences['7'] = class_sentences['7'] + ques_list[i]
                else:
                    class_sentences['7'] = ques_list[i]
    else:
        print('读取labeled_ques.xlsx数据有误')
    return class_sentences

class_sentences = getClass_sentences()
for key in class_sentences:
    print(key+":"+class_sentences[key])

# 获取数据
data = xlrd.open_workbook(file_path)
# 获取sheet
table = data.sheet_by_name('question')
# 获取总行数
nrows = table.nrows
# 获取总列数
nclos = table.ncols
for i in range(1, nrows):
    question = str(table.cell(i, 0).value).strip()
    intention = str(table.cell(i, 1).value).strip()
    if len(question) >= 1 and len(intention) >= 1:
        print(question)
        print(intention)
        print('*****************************')