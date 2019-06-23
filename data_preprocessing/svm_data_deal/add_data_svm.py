import numpy as np
import pickle

addindex = np.load('../../data_enhanced/enhanced_index/data_enhanced_index.npy')



with open('./data_cut_for_svm/data_train_fact_cut.pkl',
          mode='rb') as f:
    train_data = pickle.load(f)
with open('./data_cut_for_svm/data_train_label.pkl',
          mode='rb') as f:
    train_label = pickle.load(f)
print(len(train_data))
for i in addindex:
    train_data.append(train_data[int(i)])

print(len(train_data))
with open('./svm_enhanced_data/enhanced_data_train_fact_cut.pkl',
          mode='wb') as f:
    pickle.dump(train_data,f)