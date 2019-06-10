import csv
import re
from string import digits
from math import log
import jieba
import math
import matplotlib.pyplot as plt

class Classifier(object):
    def __init__(self):
        self.train_id = []          # train.csv:id  (id by string)
        self.train_title = {}       # train.csv: id -> title_line
        self.train_label = {}       # train.csv: id -> label

        self.idf = {}

        self.test_id = []           # test.csv: id
        self.test_title = {}        # test.csv: id -> title_line

        self.all_word = {}

    def readTrainFile(self, path):
        train_reader = csv.reader(open(path, encoding='utf-8'))
        flag = 0
        for row in train_reader:
            if flag == 0:
                flag = 1
                continue
            self.train_id.append(row[0])
            self.train_title[row[0]] = row[1] + ',' + row[2]
            self.train_label[row[0]] = row[3]
        print("READ train.csv DONE")

    def readTestFile(self, path):
        test_reader = csv.reader(open(path, encoding='utf-8'))
        flag = 0
        for row in test_reader:
            if flag == 0:
                flag = 1
                continue
            self.test_id.append(row[0])
            self.test_title[row[0]] = row[1] + ',' + row[2] 
        print("READ test.csv DONE")

    def dataPreProcess(self):
        remove_digits = str.maketrans('', '', digits)
        # remove Chinese stop words, not sure
        stop_words = ["的", "地", "得", " ", "了", "吗", "么", "吧", "呢", "罢", "呀", "啊", "啦", "哇", "嘛"]

        # train.csv
        for id in self.train_id:
            # remove number
            self.train_title[id] = self.train_title[id].translate(remove_digits)
            # remove symbol
            self.train_title[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.train_title[id])
            # to lowercase
            self.train_title[id] = self.train_title[id].lower()
            for stopword in stop_words:
                self.train_title[id] = self.train_title[id].replace(stopword, "")

        # test.csv
        for id in self.test_id:
            self.test_title[id] = self.test_title[id].translate(remove_digits)
            self.test_title[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.test_title[id])
            self.test_title[id] = self.test_title[id].lower()
            for stopword in stop_words:
                self.test_title[id] = self.test_title[id].replace(stopword, "")
        
        print("DATA Pre-Process DONE")
    
    def splitProc(self, wds):#input the string
        mp = {}
        wd = jieba.cut(wds.strip())
        for w in wd:
            if w not in mp:
                mp[w] = 1
                if not w in self.all_word:
                    self.all_word[w] = 1
            else:
                mp[w] = mp[w] + 1
        return mp

    def title2vec(self):
        for id in self.train_id:
            self.train_title[id] = self.splitProc(self.train_title[id])
        
        for id in self.test_id:
            self.test_title[id] = self.splitProc(self.test_title[id])
        print('split done')

        file = open('train_feat.txt', 'w')
        for id in self.train_id:    
            for wd in self.all_word:
                if wd in self.train_title[id]:
                    file.write(str(self.train_title[id][wd]) + ' ')
                else:
                    file.write('0' + ' ')
            file.write('\n')
            print(id)
        file.close()

        print('title2vec done')

if __name__ == "__main__":
    my_classifier = Classifier()
    my_classifier.readTrainFile("train.csv")
    my_classifier.readTestFile("test.csv")
    my_classifier.dataPreProcess()
    my_classifier.title2vec()
    #print(my_classifier.splitProc(my_classifier.train_title['0']))

    print("ALL DONE")