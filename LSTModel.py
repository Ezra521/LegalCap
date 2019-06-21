import os
import time
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D,GlobalMaxPool1D,BatchNormalization,Input,Embedding,Dense,Dropout
from get_evaluate import get_evaluate


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

isEnhanced = False

num_words = 80000 #字典的个数
maxlen = 400 #每一个输入样本的长度


kernel_size = 3
DIM = 512  #词向量的维度
batch_size = 256


n_start = 1
n_end = 11
result_list=[]

if isEnhanced:
    fact_train = np.load('./data_enhanced/enhanced_data/enhanced_data_train_fact_pad_seq_80000_400.npy')
    labels_train = np.load('./data_enhanced/enhanced_data/enhanced_data_train_labels_accusation.npy')
else:
    fact_train =np.load('./data_preprocessing/data_deal/data_model_use/fact/data_train_fact_pad_seq_80000_400.npy')
    labels_train =np.load('./data_preprocessing/data_deal/data_model_use/labels/data_train_labels_accusation.npy')


fact_test =np.load('./data_preprocessing/data_deal/data_model_use/fact/data_test_fact_pad_seq_80000_400.npy')
labels_test =np.load('./data_preprocessing/data_deal/data_model_use/labels/data_test_labels_accusation.npy')

fact_valid =np.load('./data_preprocessing/data_deal/data_model_use/fact/data_valid_fact_pad_seq_80000_400.npy')
labels_valid =np.load('./data_preprocessing/data_deal/data_model_use/labels/data_valid_labels_accusation.npy')



def get_model():
    data_input = Input(shape=[fact_train.shape[1]])
    word_vec = Embedding(input_dim=num_words + 1,
                         input_length=maxlen,
                         output_dim=DIM,
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec

    x = LSTM(128,activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation="relu")(x)
    x = Dense(labels_train.shape[1], activation="sigmoid")(x)


    #
    # x = Conv1D(filters=512, kernel_size=[kernel_size], strides=1, padding='same', activation='relu')(x)
    # x = GlobalMaxPool1D()(x)
    # x = BatchNormalization()(x)
    # x = Dense(1000, activation="relu")(x)
    # x = Dropout(0.2)(x)
    # x = Dense(labels_train.shape[1], activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    model = get_model()
    for i in range(n_start, n_end):
        model.fit(x=fact_train, y=labels_train, batch_size=batch_size,validation_data=(fact_valid,labels_valid), epochs=1, verbose=1)
        # model.fit(x=fact_test, y=labels_test, batch_size=batch_size, epochs=1, verbose=1)
        if isEnhanced:
            model.save('./model_save/LSTM_Enhanced/LSTM_epochs_%d.h5' % i)
        else:
            model.save('./model_save/LSTM_No_Enhanced/LSTM_epochs_%d.h5' % i)
        y = model.predict(fact_test[:])
        print('第%s次迭代测试结果如下：' % i)
        #获取测试集结果
        rs = get_evaluate(y_pred=y, y_true=labels_test, type="top")
        result_list.append(rs.get_all_evaluate())
        print(pd.DataFrame(result_list, columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"]).to_string(index=True))

    df =pd.DataFrame(result_list,columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"])
    nowtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    df.to_csv("./model_save/result_csv/LSTM"+ nowtime + ".csv", index=True, encoding="utf_8_sig", mode="a", header=True)
    print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('###################################################################################################################\n')
