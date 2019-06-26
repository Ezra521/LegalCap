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

clase_num_index = {i:[] for i in range(1,14)}

#按照标签的个数取得所有的个数标签对应样本的index存在字典里面备用
for n, i in enumerate(datalabel):
    clase_num_index[len(i)].append(n)
