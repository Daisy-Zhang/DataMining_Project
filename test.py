import csv

result_label = ["agreed", "disagreed", "unrelated"]

test_row_list = []

result_file = open('result.txt', 'w')

test_reader = csv.reader(open('test.csv', encoding='utf-8'))
flag = 0
for row in test_reader:
    if flag == 0:
        flag = 1
        continue
    random_value = random.randint(0,2)
    result_file.write(row[0] + "	" + result_label[random_value] + "\n")
    test_row_list.append(row)

result_file.close()