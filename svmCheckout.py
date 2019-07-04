from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics.scorer import f1_score,recall_score,precision_score
import pickle

isEnhanced = True
num = 1

def getresult(y_pred,test_label):
    f1_micro = f1_score(y_pred=y_pred, y_true=test_label, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=test_label, pos_label=1, average='macro')
    return (f1_micro,f1_macro)

if isEnhanced:
    tfidf = joblib.load('./model_save/TFIDFSVM_Enhanced/tfidf_no_enhanced.model')
    svm = joblib.load('./model_save/TFIDFSVM_Enhanced/svm_no_enhanced.model')
else:
    tfidf = joblib.load('./model_save/TFIDFSVM_No_Enhanced/tfidf_no_enhanced.model')
    svm = joblib.load('./model_save/TFIDFSVM_No_Enhanced/svm_no_enhanced.model')

with open('./data_preprocessing/svm_data_deal/svm_multi_data/fact/'+str(num)+'_num_index_test_data.npy',
          mode='rb') as f:
    test_data = pickle.load(f)
with open('./data_preprocessing/svm_data_deal/svm_multi_data/label/'+str(num)+'_num_index_test_label.npy',
          mode='rb') as f:
    test_label = pickle.load(f)
vec = tfidf.transform(test_data)

y_pred = svm.predict(vec)

f1_micro,f1_macro = getresult(y_pred,test_label)

# print("test_label",test_label)
# print("y_pred",y_pred)
print("f1_micro:",f1_micro,"\nf1_macro:",f1_macro)