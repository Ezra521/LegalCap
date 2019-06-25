import sys
import numpy as np
import pandas as pd
sys.path.append("..")#让sys目录变成父节点
from data_preprocessing.data_transform import  data_transform



def show_datanum_distribution():
    # 罪名数量分布
    labels = np.load('../data_preprocessing/data_deal/data_model_use/labels/data_all_labels_accusation.npy')
    x = labels.sum(axis=1)
    for i in range(int(x.max()) + 1):
        # s1 = (x==i)
        # print('罪名数=%d的有%d个' % (i, s1.sum()))
        print('罪名数=%d的有%d个' % (i, (x == i).sum()))
        #一个案件共有几个罪名的统计  列出来

# show_datanum_distribution()

original_dataname = "data_train"
data_transform = data_transform()

data_transform.read_data(path="../data_preprocessing/data_original/"+ original_dataname +".json")
data_transform.extract_data(name='accusation')
datalabel = data_transform.extraction['accusation']

class_name = []
class_index = []

for n, i in enumerate(datalabel):
    # 只取罪名只有一个的样本  然后 获取罪名 和索引 索引就是第几个
    if len(i) == 1:
        class_name.append(i[0])
        class_index.append(n)

# 把罪名变成array数组
class_name_array = np.array(class_name)
# 把索引变成array数组
class_index_array = np.array(class_index)

accusation_set = list(set(class_name))

accu_num = {}

for i in accusation_set:
    accu_num[i] = class_name.count(i)
    class_index_i = class_index_array[class_name_array == i]
    np.save('./all_accu_num/%s_%d.npy'%(i,accu_num[i]),class_index_i)


print(accu_num)

a = sorted(accu_num.items(), key=lambda x: x[1], reverse=True)
for i in a:
    print(i)