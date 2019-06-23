from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics.scorer import f1_score,recall_score,precision_score
import pickle

original_dataname = "data_test"
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

def saveModel():
    print("reading data...")

    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/data_train_fact_cut.pkl',
              mode='rb') as f:
        train_data = pickle.load(f)
    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/data_train_label.pkl',
              mode='rb') as f:
        train_label = pickle.load(f)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)
    vec = tfidf.transform(train_data)
    # vecarray = vec.toarray()
    print('train SVC')
    svm = train_SVC(vec, train_label)
    joblib.dump(tfidf, './model_save/TFIDFSVM_No_Enhanced/tfidf_no_enhanced.model')
    joblib.dump(svm, './model_save/TFIDFSVM_No_Enhanced/svm_no_enhanced.model')
    print("finsh svc training")

def getresult(y_pred,test_label):
    f1_micro = f1_score(y_pred=y_pred, y_true=test_label, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=test_label, pos_label=1, average='macro')
    return (f1_micro,f1_macro)

if __name__ == "__main__":
    # saveModel()
    tfidf = joblib.load('./model_save/TFIDFSVM_No_Enhanced/tfidf_no_enhanced.model')
    svm = joblib.load('./model_save/TFIDFSVM_No_Enhanced/svm_no_enhanced.model')
    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/' + original_dataname + '_fact_cut.pkl',
              mode='rb') as f:
        test_data = pickle.load(f)
    with open('./data_preprocessing/svm_data_deal/data_cut_for_svm/' + original_dataname + '_label.pkl',
              mode='rb') as f:
        test_label = pickle.load(f)
    vec = tfidf.transform(test_data)

    y_pred = svm.predict(vec)

    f1_micro,f1_macro = getresult(y_pred,test_label)

    # print("test_label",test_label)
    # print("y_pred",y_pred)
    print("f1_micro:",f1_micro,"\nf1_macro:",f1_macro)