import numpy as np
import pickle
import os
# addindex = np.load('../../data_enhanced/enhanced_index/data_enhanced_index.npy')



with open('./data_cut_for_svm/data_test_fact_cut.pkl',mode='rb') as f:
    train_data = pickle.load(f)
with open('./data_cut_for_svm/data_test_label.pkl',mode='rb') as f:
    train_label = pickle.load(f)
print(len(train_data))

fact = np.array(train_data)
label = np.array(train_label)

for root,dirs,files in os.walk("./multi_label_data_index"):
    # print(root,dirs,files)
    for i in files:
        index = np.load(os.path.join(root,i))
        # print(type(index))

        if index.size > 0:
            x = fact[index]
            y = label[index]
            x = x.tolist()
            y = y.tolist()
            with open("./svm_multi_data/fact/"+i.split(".")[0]+"_data.npy",
                      mode='wb') as f:
                pickle.dump(x, f)

            with open("./svm_multi_data/label/"+i.split(".")[0]+"_label.npy",
                      mode='wb') as f:
                pickle.dump(y, f)
            # np.save("./svm_multi_data/fact/"+i.split(".")[0]+"_data.npy", x)
            # np.save("./svm_multi_data/label/"+i.split(".")[0]+"_label.npy",y)


# for i in addindex:
#     train_data.append(train_data[int(i)])
#     train_label.append(train_label[int(i)])
#
# print(len(train_data))
#
# with open('./svm_multi_data/enhanced_data_train_fact_cut.pkl',
#           mode='wb') as f:
#     pickle.dump(train_data,f)
#
# with open('./svm_multi_data/enhanced_data_train_label.pkl',
#           mode='wb') as f:
#     pickle.dump(train_label,f)