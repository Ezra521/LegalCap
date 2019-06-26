import numpy as np
import os
fact = np.load('../data_preprocessing/data_deal/data_model_use/fact/data_train_fact_pad_seq_80000_400.npy')
label = np.load('../data_preprocessing/data_deal/data_model_use/labels/data_train_labels_accusation.npy')

# index = np.load('./all_accu_num/2_徇私舞弊不征、少征税款.npy')
# x = fact[index]
# y = label[index]
# np.save("./one_label_data/2_徇私舞弊不征、少征税款.npy",x)
# np.save('./one_label_data/2_徇私舞弊不征、少征税款_label.npy',y)
# os.walk()
for root,dirs,files in os.walk("./one_label_data_index"):
    print(root,dirs,files)
    for i in files:
        index = np.load(os.path.join(root,i))
        x = fact[index]
        y = label[index]
        np.save("./one_label_data/fact/"+i.split(".")[0]+"_data.npy", x)
        np.save("./one_label_data/label/"+i.split(".")[0]+"_label.npy",y)