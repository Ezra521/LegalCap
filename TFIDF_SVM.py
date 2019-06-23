from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle

original_dataname = "data_train"
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
    print(type(vec),type(label))
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
    # vecarray = vec.toarray()
    print('accu SVC')
    svm = train_SVC(vec, train_label)
    joblib.dump(tfidf, './model_save/TFIDFSVM_No_Enhanced/tfidf_no_enhanced.model')
    joblib.dump(svm, './model_save/TFIDFSVM_No_Enhanced/svm_no_enhanced.model')


    # y = accu.predict(vec)