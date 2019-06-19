import numpy as np


addindex = np.load('../data_preprocessing/data_deal/index_add_accusation_2000_10.npy')
fact = np.load('../data_preprocessing/data_deal/fact_pad_seq/big_fact_pad_seq_80000_400.npy')
label = np.load('../data_preprocessing/data_deal/labels/big_labels_accusation.npy')

# addfact = fact + fact[addindex]
addfact = np.append(fact,fact[addindex],axis=0)
addlabel = np.append(label,label[addindex],axis=0)
print(len(addindex))
print(len(fact))
print(len(label))
print(len(addfact),addfact,type(addfact))
print(len(addlabel),addlabel,type(addlabel))

np.save("../data_preprocessing/data_deal/fact_pad_seq/add_big_fact_pad_seq_80000_400.npy",addfact)
np.save('../data_preprocessing/data_deal/labels/add_big_labels_accusation.npy',addlabel)
