from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
# from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac

original_dataname = "test"
dim = 5000



with open('.data_preprocessing/svm_data_deal/data_cut_for_svm/'+original_dataname+'_fact_cut_11.pkl'%len(alltext),mode='wb') as f:
    traindata = pickle.load(f)
with open('.data_preprocessing/svm_data_deal/data_cut_for_svm/'+original_dataname+'_label_%d.pkl'%len(alltext),mode='wb') as f:
    pickle.load(f)




def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf

def train_SVC(vec, label):
	SVC = LinearSVC()
	SVC.fit(vec, label)
	return SVC

def readdata():
    with open('./data_deal/data_cut/' + original_dataname + '_fact_cut_0_100000.pkl', mode='rb') as f:
        fact_cut = pickle.load(f)
    return fact_cut
if __name__ == "__main__":
    print("reading data...")
    # alltext =