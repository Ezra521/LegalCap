import numpy as np


addindex = np.load('./enhanced_index/data_enhanced_index.npy')
fact = np.load('../data_preprocessing/data_deal/data_model_use/fact/data_train_fact_pad_seq_80000_400.npy')
label = np.load('../data_preprocessing/data_deal/data_model_use/labels/data_train_labels_accusation.npy')

# addfact = fact + fact[addindex]
addfact = np.append(fact,fact[addindex],axis=0)
addlabel = np.append(label,label[addindex],axis=0)
print(len(addindex))
print(len(fact))
print(len(label))
print(len(addfact),addfact,type(addfact))
print(len(addlabel),addlabel,type(addlabel))

np.save("./enhanced_data/enhanced_data_train_fact_pad_seq_80000_400.npy",addfact)
np.save('./enhanced_data/enhanced_data_train_labels_accusation.npy',addlabel)
