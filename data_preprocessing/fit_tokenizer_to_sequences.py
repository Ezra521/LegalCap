from data_transform import data_transform
import pickle
import jieba
# import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

jieba.setLogLevel('WARN')

num_words = 80000
maxlen = 400

# original_dataname = "data_valid"
original_dataname = "data_train"

data_transform = data_transform()
data_transform.read_data(path="./data_original/"+ original_dataname +".json")

data_transform.extract_data(name='fact')
originaldatanum = len(data_transform.extraction["fact"])
each_pkl_data_num = 100000

tokenizer_fact = Tokenizer(num_words=num_words)

# train tokenizer
def train_tokenizer(tokenizer_fact):
    # tokenizer_fact = Tokenizer(num_words=num_words)
    for i in range(int(originaldatanum/each_pkl_data_num)+1):
        if i < int(originaldatanum / each_pkl_data_num):
            print('start train_tokenizer_' + original_dataname + '_fact_cut_%d_%d' % (
            i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num))
            with open('./data_deal/data_cut/'+ original_dataname +'_fact_cut_%d_%d.pkl' % (
            i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num), mode='rb') as f:
                big_fact_cut = pickle.load(f)
            texts_cut_len = len(big_fact_cut)  # 共多少条记录
            n = 0
            # 分批训练，
            while n < texts_cut_len:
                tokenizer_fact.fit_on_texts(texts=big_fact_cut[n:n + 10000])
                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            # pass
        else:
            # pass
            print('start train_tokenizer'+ original_dataname +'_fact_cut_%d_%d' % (i * each_pkl_data_num,originaldatanum))
            with open('./data_deal/data_cut/'+ original_dataname +'_fact_cut_%d_%d.pkl' % (i * each_pkl_data_num,originaldatanum), mode='rb') as f:
                big_fact_cut = pickle.load(f)
            texts_cut_len = len(big_fact_cut)#共多少条记录
            n = 0
            # 分批训练，
            while n < texts_cut_len:
                tokenizer_fact.fit_on_texts(texts=big_fact_cut[n:n + 10000])
                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            # print('finish big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
        # print(tokenizer_fact.word_counts)

    with open('../model_save/tokenizer/tokenizer_fact_%d.pkl' % (num_words), mode='wb') as f:
        pickle.dump(tokenizer_fact, f)

    # with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
    #     tokenizer_fact=pickle.load(f)
        # print(tokenizer_fact.word_counts)

# train tokenizer and save tokenizer
# train_tokenizer(tokenizer_fact)


with open('../model_save/tokenizer/tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
    tokenizer_fact=pickle.load(f)

# texts_to_sequences
def texts_to_sequences(tokenizer_fact):
    for i in range(int(originaldatanum/each_pkl_data_num)+1):
        if i < int(originaldatanum / each_pkl_data_num):
            # pass

            print(
                'start text_to_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num))
            with open(
                './data_deal/data_cut/' + original_dataname + '_fact_cut_%d_%d.pkl' % (i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num),
                mode='rb') as f:
                fact_cut = pickle.load(f)
            print(len(fact_cut))
            # 分批执行 texts_to_sequences
            fact_seq = tokenizer_fact.texts_to_sequences(texts=fact_cut)
            # print(big_fact_seq)
            with open(
                './data_deal/fact_seq/' + original_dataname + '_fact_seq_%d_%d.pkl' % (i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num),
                mode='wb') as f:
                pickle.dump(fact_seq, f)
            print(
                'finish text_to_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * each_pkl_data_num, i * each_pkl_data_num + each_pkl_data_num))
        else:
            # pass
            print('start text_to_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * each_pkl_data_num, originaldatanum))
            with open('./data_deal/data_cut/' + original_dataname + '_fact_cut_%d_%d.pkl' % (i * each_pkl_data_num, originaldatanum), mode='rb') as f:
                fact_cut = pickle.load(f)
            # print(len(fact_cut))
            # 分批执行 texts_to_sequences
            fact_seq = tokenizer_fact.texts_to_sequences(texts=fact_cut)
            # print(big_fact_seq)
            with open('./data_deal/fact_seq/' + original_dataname + '_fact_seq_%d_%d.pkl' % (i * each_pkl_data_num, originaldatanum), mode='wb') as f:
                pickle.dump(fact_seq, f)
            print('finish text_to_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * each_pkl_data_num, originaldatanum))

texts_to_sequences(tokenizer_fact)

# pad_sequences
def pad_seq():
    for i in range(int(originaldatanum/each_pkl_data_num)+1):

        if i < int(originaldatanum / each_pkl_data_num):
            # pass
            print('start pad_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
            with open('./data_deal/fact_seq/' + original_dataname + '_fact_seq_%d_%d.pkl' % (i * 100000, i * 100000 + 100000), mode='rb') as f:
                big_fact_seq = pickle.load(f)
            texts_cut_len = len(big_fact_seq)
            n = 0
            fact_pad_seq = []
            # 分批执行pad_sequences
            while n < texts_cut_len:
                #keras的方法
                '''
                pad_sequences例子
                list_1 = [[2,3,4]]
                keras.preprocessing.sequence.pad_sequences(list_1, maxlen=10)
                array([[0, 0, 0, 0, 0, 0, 0, 2, 3, 4]], dtype=int32)
                '''
                fact_pad_seq += list(pad_sequences(big_fact_seq[n:n + 20000], maxlen=maxlen,padding='post', value=0, dtype='int'))
                n += 20000
                if n < texts_cut_len:
                    print('finish pad_sequences %d samples' % n)
                else:
                    print('finish pad_sequences %d samples' % texts_cut_len)
            with open('./data_deal/fact_pad_seq/' + original_dataname + '_fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * 100000, i * 100000 + 100000),mode='wb') as f:
                pickle.dump(fact_pad_seq, f)
        else:
            print('start pad_sequences ' + original_dataname + '_fact_cut_%d_%d' % (i * 100000, originaldatanum))
            with open(
                './data_deal/fact_seq/' + original_dataname + '_fact_seq_%d_%d.pkl' % (i * 100000, originaldatanum),
                mode='rb') as f:
                big_fact_seq = pickle.load(f)
            texts_cut_len = len(big_fact_seq)
            n = 0
            fact_pad_seq = []
            # 分批执行pad_sequences
            while n < texts_cut_len:
                # keras的方法
                '''
                pad_sequences例子
                list_1 = [[2,3,4]]
                keras.preprocessing.sequence.pad_sequences(list_1, maxlen=10)
                array([[0, 0, 0, 0, 0, 0, 0, 2, 3, 4]], dtype=int32)
                '''
                fact_pad_seq += list(
                    pad_sequences(big_fact_seq[n:n + 20000], maxlen=maxlen, padding='post', value=0, dtype='int'))
                n += 20000
                if n < texts_cut_len:
                    print('finish pad_sequences %d samples' % n)
                else:
                    print('finish pad_sequences %d samples' % texts_cut_len)
            with open('./data_deal/fact_pad_seq/' + original_dataname + '_fact_pad_seq_%d_%d_%d.pkl' % (
            maxlen, i * 100000, originaldatanum), mode='wb') as f:
                pickle.dump(fact_pad_seq, f)
pad_seq()


def summary():
    fact_pad_seq = []
    for i in range(int(originaldatanum/each_pkl_data_num)+1):
        if i < int(originaldatanum / each_pkl_data_num):
            print('start ' + original_dataname + '_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
            with open('./data_deal/fact_pad_seq/' + original_dataname + '_fact_pad_seq_%d_%d_%d.pkl' % (
            maxlen, i * 100000, i * 100000 + 100000), mode='rb') as f:
                fact_pad_seq += pickle.load(f)
        else:
            print('start ' + original_dataname + '_fact_cut_%d_%d' % (i * 100000, originaldatanum))
            with open('./data_deal/fact_pad_seq/' + original_dataname + '_fact_pad_seq_%d_%d_%d.pkl' % (
            maxlen, i * 100000, originaldatanum), mode='rb') as f:
                fact_pad_seq += pickle.load(f)

    fact_pad_seq = np.array(fact_pad_seq)
    np.save('./data_deal/data_model_use/fact/' + original_dataname + '_fact_pad_seq_%d_%d.npy' % (num_words, maxlen), fact_pad_seq)

summary()