from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac

dim = 5000

original_dataname = "data_train"

def cut_text(alltext):
    count = 0
    cut = thulac.thulac(seg_only=True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(cut.cut(text, text=True))

    return train_text


def read_trainData(path):
    fin = open(path, 'r', encoding='utf8')
    alltext = []
    accu_label = []
    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d))
        line = fin.readline()
    fin.close()
    return alltext, accu_label

print("reading ...")
alltext,accu_label = read_trainData("../data_original/"+original_dataname+".json")
print('cut text...')
train_data = cut_text(alltext)

with open('./fact_cut/'+original_dataname+'_fact_cut_%d_%d.pkl',)


print('train tfidf...')
tfidf = train_tfidf(train_data)
train_data = cut_text(alltext)