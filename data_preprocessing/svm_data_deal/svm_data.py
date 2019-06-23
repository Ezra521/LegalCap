import sys
import pickle
sys.path.append("..")#让sys目录变成父节点
from data_preprocessing.data_transform import  data_transform



original_dataname = "data_train"
data_transform = data_transform()

# 读取json文件,1710857行
data_transform.read_data(path="../data_original/"+ original_dataname +".json")

data_transform.extract_data(name='fact')

originaldatanum = len(data_transform.extraction["fact"])

texts = data_transform.extraction['fact'][:]
fact_cut = data_transform.cut_texts(texts=texts, word_len=1, need_cut=True)
with open('./fact_cut/' + original_dataname + 'num_%d.pkl'% originaldatanum, mode='wb') as f:
    pickle.dump(fact_cut,f)
print("finish %d sample"%originaldatanum)