from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac

dim = 5000

original_dataname = "test"

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

with open('./data_cut_for_svm/'+original_dataname+'_fact_cut_%d.pkl'%len(alltext),mode='wb') as f:
    pickle.dump(train_data,f)
with open('./data_cut_for_svm/'+original_dataname+'_label_%d.pkl'%len(alltext),mode='wb') as f:
    pickle.dump(accu_label,f)


print("finish data cut")