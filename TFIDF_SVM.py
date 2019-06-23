from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
# from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac

original_dataname = "test"
dim = 5000

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


if __name__ == "__main__":
    print("reading data...")

    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/' + original_dataname + '_fact_cut.pkl',
              mode='rb') as f:
        train_data = pickle.load(f)
    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/' + original_dataname + '_label.pkl', mode='rb') as f:
        train_label = pickle.load(f)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)
    vec = tfidf.transform(train_data)
    print('accu SVC')
    accu = train_SVC(vec, train_label)
    joblib.dump(accu, './model_save/TFIDFSVM_No_Enhanced/svm.model')
    # alltext =