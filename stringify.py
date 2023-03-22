from functools import reduce
import os
import csv

string = "1"
result =list(map(lambda n: int(n) if n !=  '' else 0, string.split(",")))
print(result)

numeros = [1, 2, 3, 4, 5, 6]

result =reduce(lambda a,b: a+b, numeros)
print(result)

report = {
    'label1':{
        'precision':0.5,
        'recall':1.0,
        'f1-score':0.67,
        'support':1
    },
    'label2':{

    }
}

file_path = os.path.join("report_test.csv")
f = open(file_path, "w")
w = csv.writer(f)


cols = ['label', 'precision', 'recall', 'f1-score', 'support']
w.writerow(cols)

for label in report.keys():
    report[label][label]= label
    w.writerow(report[label].values())
        
    


f.close()