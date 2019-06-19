from data_transform import data_transform
import pickle
import jieba

jieba.setLogLevel('WARN')

num_words = 40000
maxlen = 400

original_dataname = "data_train"

########################################################################################
# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path="./data_original/"+ original_dataname +".json")

# 提取需要信息
data_transform_big.extract_data(name='fact')

originaldatanum = len(data_transform_big.extraction["fact"])
each_pkl_data_num = 100000

for i in range(int(originaldatanum/each_pkl_data_num)+1):
    if i < int(originaldatanum/each_pkl_data_num):
        texts = data_transform_big.extraction['fact'][i*each_pkl_data_num:(i*each_pkl_data_num + each_pkl_data_num)]
        big_fact_cut = data_transform_big.cut_texts(texts=texts, word_len=1,need_cut=True)
        with open('./data_deal/data_cut/'+original_dataname+'_fact_cut_%d_%d.pkl' % (i*each_pkl_data_num, i*each_pkl_data_num + each_pkl_data_num), mode='wb') as f:
            pickle.dump(big_fact_cut, f)
        print('finish '+original_dataname+'_fact_cut_%d_%d' % (i*each_pkl_data_num, i*each_pkl_data_num + each_pkl_data_num))
    else:
        texts = data_transform_big.extraction['fact'][i * each_pkl_data_num:(originaldatanum)]
        big_fact_cut = data_transform_big.cut_texts(texts=texts, word_len=1, need_cut=True)
        with open('./data_deal/data_cut/' +original_dataname +'_fact_cut_%d_%d.pkl' % (i * each_pkl_data_num,originaldatanum ),mode='wb') as f:
            pickle.dump(big_fact_cut, f)
        print('finish ' + original_dataname + '_fact_cut_%d_%d' % (i * each_pkl_data_num, originaldatanum))
    print("len:",len(big_fact_cut))
# print("finish "+str(originaldatanum)+" sample")
