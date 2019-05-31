import csv
import re
from string import digits

class Classifier(object):
    def __init__(self):
        self.train_id = []          # train.csv:id  (id by string)
        self.train_title1 = {}      # train.csv: id -> title1
        self.train_title2 = {}      # train.csv: id -> title2
        self.train_label = {}       # train.csv: id -> label

        self.test_id = []           # test.csv: id
        self.test_title1 = {}       # test.csv: id -> title1
        self.test_title2 = {}       # test.csv: id -> title2

        self.result_label = ["agreed", "disagreed", "unrelated"]

    def readTrainFile(self, path):
        train_reader = csv.reader(open(path, encoding='utf-8'))
        flag = 0
        for row in train_reader:
            if flag == 0:
                flag = 1
                continue
            self.train_id.append(row[0])
            self.train_title1[row[0]] = row[1]
            self.train_title2[row[0]] = row[2]
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
            self.test_title1[row[0]] = row[1]
            self.test_title2[row[0]] = row[2]
        print("READ test.csv DONE")

    def dataPreProcess(self):
        remove_digits = str.maketrans('', '', digits)
        # remove Chinese stop words, not sure
        stop_words = ["的", "地", "得", " "]

        # train.csv
        for id in self.train_id:
            # remove number
            self.train_title1[id] = self.train_title1[id].translate(remove_digits)
            # remove symbol
            self.train_title1[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.train_title1[id])
            # to lowercase
            self.train_title1[id] = self.train_title1[id].lower()
            for stopword in stop_words:
                self.train_title1[id].replace(stopword, "")

            self.train_title2[id] = self.train_title2[id].translate(remove_digits)
            self.train_title2[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.train_title2[id])
            self.train_title2[id] = self.train_title2[id].lower()
            for stopword in stop_words:
                self.train_title2[id].replace(stopword, "")

        # test.csv
        for id in self.test_id:
            self.test_title1[id] = self.test_title1[id].translate(remove_digits)
            self.test_title1[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.test_title1[id])
            self.test_title1[id] = self.test_title1[id].lower()
            for stopword in stop_words:
                self.test_title1[id].replace(stopword, "")

            self.test_title2[id] = self.test_title2[id].translate(remove_digits)
            self.test_title2[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.test_title2[id])
            self.test_title2[id] = self.test_title2[id].lower()
            for stopword in stop_words:
                self.test_title2[id].replace(stopword, "")
        
        print("DATA Pre-Process DONE")

if __name__ == "__main__":
    my_classifier = Classifier()
    my_classifier.readTrainFile("train.csv")
    my_classifier.readTestFile("test.csv")
    my_classifier.dataPreProcess()
    for i in my_classifier.train_id:
        print(my_classifier.train_title1[i])
    print("ALL DONE")