import os
import time
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import GRU, Bidirectional,Flatten,SpatialDropout1D,Input,Embedding,Dense,Dropout
from get_evaluate import get_evaluate
from Capsule_Keras import Capsule


#选择训练显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

num_words = 80000 #字典的个数
maxlen = 400 #每一个输入样本的长度
kernel_size = 3
DIM = 400  #词向量的维度
batch_size = 256
gru_len = 128
Routings = 4
Num_capsule = 10
Dim_capsule = 18
dropout_p = 0.2
rate_drop_dense = 0.1

n_start = 1
n_end = 51
result_list=[]


isEnhanced = True

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
    x = SpatialDropout1D(rate_drop_dense)(word_vec)
    x = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(x)
    # x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    x = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings )(x)
    x = Flatten()(x)
    # x = Dropout(dropout_p)(x)
    # x = Dense(500, activation='sigmoid')(x)
    # x = Dense(300, activation='sigmoid')(x)
    output = Dense(202, activation='sigmoid')(x)
    model = Model(inputs=data_input, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    model = get_model()
    for i in range(n_start, n_end):
        model.fit(x=fact_train, y=labels_train, batch_size=batch_size, validation_data=(fact_valid, labels_valid),
                  epochs=1, verbose=1)
        if isEnhanced:
            model.save('./model_save/Capsule_Enhanced/Capsule_epochs_%d.h5' % i)
        else:
            model.save('./model_save/Capsule_No_Enhanced/Capsule_epochs_%d.h5' % i)
        y = model.predict(fact_test[:])
        print('第%s次迭代测试结果如下：' % i)
        #获取测试集结果
        rs = get_evaluate(y_pred=y, y_true=labels_test, type="top")
        result_list.append(rs.get_all_evaluate())
        print(pd.DataFrame(result_list, columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"]).to_string(index=True))

    df =pd.DataFrame(result_list,columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"])
    nowtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if isEnhanced:
        df.to_csv("./model_save/result_csv/enhanced_Capsule"+ nowtime + ".csv", index=True, encoding="utf_8_sig", mode="a", header=True)
    else:
        df.to_csv("./model_save/result_csv/no_enhanced_Capsule"+ nowtime + ".csv", index=True, encoding="utf_8_sig", mode="a", header=True)
    print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('###################################################################################################################\n')
