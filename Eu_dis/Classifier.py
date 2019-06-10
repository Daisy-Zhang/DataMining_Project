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
        self.train_title1 = {}      # train.csv: id -> title1
        self.train_title2 = {}      # train.csv: id -> title2
        self.train_label = {}       # train.csv: id -> label

        self.test_id = []           # test.csv: id
        self.test_title1 = {}       # test.csv: id -> title1
        self.test_title2 = {}       # test.csv: id -> title2

        self.idf = {}
        self.tot = 0

        self.train_unrelated_dis = 0.0
        self.train_agreed_dis = 0.0
        self.train_disagreed_dis = 0.0

        self.unrelated_thre = 0.0
        self.agreed_thre = 0.0
        self.disagreed_thre = 0.0

        self.agreed_dict = {}
        self.disagreed_dict = {}

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
        stop_words = ["的", "地", "得", " ", "了", "吗", "么", "吧", "呢", "罢", "呀", "啊", "啦", "哇", "嘛"]

        # train.csv
        for id in self.train_id:
            # remove number
            self.train_title1[id] = self.train_title1[id].translate(remove_digits)
            # remove symbol
            self.train_title1[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.train_title1[id])
            # to lowercase
            self.train_title1[id] = self.train_title1[id].lower()
            for stopword in stop_words:
                self.train_title1[id] = self.train_title1[id].replace(stopword, "")

            self.train_title2[id] = self.train_title2[id].translate(remove_digits)
            self.train_title2[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.train_title2[id])
            self.train_title2[id] = self.train_title2[id].lower()
            for stopword in stop_words:
                self.train_title2[id] = self.train_title2[id].replace(stopword, "")

        # test.csv
        for id in self.test_id:
            self.test_title1[id] = self.test_title1[id].translate(remove_digits)
            self.test_title1[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.test_title1[id])
            self.test_title1[id] = self.test_title1[id].lower()
            for stopword in stop_words:
                self.test_title1[id] = self.test_title1[id].replace(stopword, "")

            self.test_title2[id] = self.test_title2[id].translate(remove_digits)
            self.test_title2[id] = re.sub("[+\.\“”： \t《》!\/_,$%^*()+\"\']+|[+——:;；?！，。？、~@#￥%……&*（）-]+", "", self.test_title2[id])
            self.test_title2[id] = self.test_title2[id].lower()
            for stopword in stop_words:
                self.test_title2[id] = self.test_title2[id].replace(stopword, "")
        
        print("DATA Pre-Process DONE")

    def StringProcess(self, wds):#input the string
        mp = {}
        wd = jieba.cut(wds.strip())
        for w in wd:
            if not w in mp:
                mp[w] = 1
                if not w in self.idf:
                    self.idf[w] = 1
                else:
                    self.idf[w] = self.idf[w] + 1
            else:
                mp[w] = mp[w] + 1
        return [wds, mp]	#return the tuple (string, tf vector of the string)

    def TfIdfCalculate(self):
        for id in self.train_id:
            self.tot = self.tot + 1
            self.train_title1[id] = self.StringProcess(self.train_title1[id])
            self.tot = self.tot + 1
            self.train_title2[id] = self.StringProcess(self.train_title2[id])
        for id in self.test_id:
            self.tot = self.tot + 1
            self.test_title1[id] = self.StringProcess(self.test_title1[id])
            self.tot = self.tot + 1
            self.test_title2[id] = self.StringProcess(self.test_title2[id])

        for wd in self.idf:
            self.idf[wd] = log(self.tot / self.idf[wd])
        print("IDF Calculate Finished")
        for id in self.train_id:
            for wd in self.train_title1[id][1]:
                self.train_title1[id][1][wd] *= self.idf[wd]
            for wd in self.train_title2[id][1]:
                self.train_title2[id][1][wd] *= self.idf[wd]
        for id in self.test_id:
            for wd in self.test_title1[id][1]:
                self.test_title1[id][1][wd] *= self.idf[wd]
            for wd in self.test_title2[id][1]:
                self.test_title2[id][1][wd] *= self.idf[wd]
        print ("TF_IDF Calculate Finished")
        
    def calEuDis(self, dict1, dict2):
        tmp = 0
        tc = 0
        dictc = {}
        for wd1 in dict1:
            if wd1 in dict2:
                tmp += math.pow(dict1[wd1] - dict2[wd1], 2)
                tc = abs(dict1[wd1] - dict2[wd1])
            else:
                tmp += math.pow(dict1[wd1], 2)
                tc = dict1[wd1]
            if wd1 in dictc:
                dictc[wd1] += tc
            else:
                dictc[wd1] = tc
        
        for wd2 in dict2:
            if wd2 not in dict1:
                tmp += math.pow(dict2[wd2], 2)
                tc = dict2[wd2]
                if wd2 in dictc:
                    dictc[wd2] += tc
                else:
                    dictc[wd2] = tc

        return (dictc,math.sqrt(tmp))
    
    def calCosDis(self, dict1, dict2):
        a = 0
        b = 0
        c = 0
        for wd1 in dict1:
            if wd1 in dict2:
                a += dict1[wd1] * dict2[wd1]

        for wd1 in dict1:
            b += dict1[wd1] * dict1[wd1]
        b = math.sqrt(b)

        for wd2 in dict2:
            c += dict2[wd2] * dict2[wd2]
        c = math.sqrt(c)

        if b == 0 or c == 0:
            #print("division by zero")
            return 0

        return a / (b * c)

    def selfLearn(self, tar_dict, step, tar_ratio, init_val, tot):
        tmp_ratio = 0
        ans = 0
        threshold = init_val
        while tmp_ratio <= tar_ratio:
            ans = 0
            for dis in tar_dict:
                if dis > threshold:
                    ans += tar_dict[dis]
            tmp_ratio = ans / tot
            threshold -= step

        print(threshold)
        return threshold

    def trainDisProcess(self):
    	# distance_value -> times
        dis_times = {}
        agreed_times = {}
        disagreed_times = {}
        unrelated_times = {}
        train_tot = 0
        agreed_tot = 0
        disagreed_tot = 0
        unrelated_tot = 0
        for i in self.train_id:
            train_tot += 1
            # Cos dis
            '''if self.train_label[i] == 'agreed':
                self.train_agreed_dis += self.calCosDis(self.train_title1[i][1], self.train_title2[i][1])

            elif self.train_label[i] == 'disagreed':
                self.train_disagreed_dis += self.calCosDis(self.train_title1[i][1], self.train_title2[i][1])

            elif self.train_label[i] == 'unrelated':
                self.train_unrelated_dis += self.calCosDis(self.train_title1[i][1], self.train_title2[i][1])'''
            # Eu dis
            (dc,tmp) = self.calEuDis(self.train_title1[i][1], self.train_title2[i][1])
            if tmp not in dis_times:
            	dis_times[tmp] = 1
            else:
            	dis_times[tmp] += 1

            if self.train_label[i] == 'agreed':
                self.train_agreed_dis += tmp
                agreed_tot += 1
                if tmp not in agreed_times:
                    agreed_times[tmp] = 1
                else:
                    agreed_times[tmp] += 1
                for wd in dc:
                    if not wd in self.agreed_dict:
                        self.agreed_dict[wd] = dc[wd]
                    else:
                        self.agreed_dict[wd] += dc[wd]

            elif self.train_label[i] == 'disagreed':
                self.train_disagreed_dis += tmp
                disagreed_tot += 1
                if tmp not in disagreed_times:
                    disagreed_times[tmp] = 1
                else:
                    disagreed_times[tmp] += 1
                for wd in dc:
                    if not wd in self.disagreed_dict:
                        self.disagreed_dict[wd] = dc[wd]
                    else:
                        self.disagreed_dict[wd] += dc[wd]

            elif self.train_label[i] == 'unrelated':
                self.train_unrelated_dis += tmp
                unrelated_tot += 1
                if tmp not in unrelated_times:
                    unrelated_times[tmp] = 1
                else:
                    unrelated_times[tmp] += 1
        self.train_agreed_dis = self.train_agreed_dis / agreed_tot
        self.train_disagreed_dis = self.train_disagreed_dis / disagreed_tot
        self.train_unrelated_dis = self.train_unrelated_dis / unrelated_tot

        for wd in self.agreed_dict:
            self.agreed_dict[wd] /= agreed_tot
        for wd in self.disagreed_dict:
            self.disagreed_dict[wd] /= disagreed_tot
        #print(dis_times)
        '''plt.figure(1)
        for dis in dis_times:
            plt.plot(dis, dis_times[dis])
        plt.show()'''
        #print(self.train_agreed_dis)     #cos: 0.13005  Eu: 22
        #print(self.train_disagreed_dis)  #cos: 0.01019  Eu: 23
        #print(self.train_unrelated_dis)  #cos: 0.09972  Eu: 29

        self.unrelated_thre = self.selfLearn(unrelated_times, 0.5, 0.95, self.train_unrelated_dis, unrelated_tot)
        self.disagreed_thre = self.selfLearn(disagreed_times, 0.5, 0.95, self.train_disagreed_dis, disagreed_tot)
        self.agreed_thre = self.selfLearn(agreed_times, 0.5, 0.95, self.train_agreed_dis, agreed_tot)

        print("train Cos Dis Pro Done")

    def getResult(self):
        result_file = open('result.txt', 'w')
        for i in self.test_id:
            # Cos
            '''tmp_dis = self.calCosDis(self.test_title1[i][1], self.test_title2[i][1])
            if tmp_dis <= self.train_disagreed_dis:
                result_file.write(i + "	" + "disagreed" + "\n")
            elif tmp_dis <= self.train_agreed_dis:
                result_file.write(i + "	" + "unrelated" + "\n")
            else:
                result_file.write(i + "	" + "agreed" + "\n")'''
            # Eu
            (tmpdc,tmp_dis) = self.calEuDis(self.test_title1[i][1], self.test_title2[i][1])
            if tmp_dis >= self.unrelated_thre:
                result_file.write(i + "\t" + "unrelated" + "\n")
            else:
                (dict_a,dis_a) = self.calEuDis(tmpdc, self.agreed_dict)
                (dict_d,dis_d) = self.calEuDis(tmpdc, self.disagreed_dict)
                if dis_a < dis_d:
                	result_file.write(i + "\t" + "agreed" + "\n")
                else:
                    result_file.write(i + "\t" + "disagreed" + "\n")

        result_file.close()
        print("get Result Done")

if __name__ == "__main__":
    my_classifier = Classifier()
    my_classifier.readTrainFile("../DM_project_data/train.csv")
    my_classifier.readTestFile("../DM_project_data/test.csv")
    my_classifier.dataPreProcess()
    my_classifier.TfIdfCalculate()
    my_classifier.trainDisProcess()
    my_classifier.getResult()

    print("ALL DONE")