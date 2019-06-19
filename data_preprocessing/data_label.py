from data_transform import data_transform
import json
import pickle
import thulac
import jieba
import numpy as np

jieba.setLogLevel('WARN')

num_words = 40000
maxlen = 400

# original_dataname = "data_valid"
# original_dataname = "data_valid"
original_dataname = "data_train"


########################################################################################
# 数据集处理
data_transform = data_transform()

# 读取json文件
data_transform.read_data(path="./data_original/"+ original_dataname +".json")

# 创建数据one-hot标签
data_transform.extract_data(name='accusation')
# data_transform.extraction['accusation']
# print(data_transform.extraction['accusation'])
#[['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['妨害公务', '故意伤害', '盗窃']]
data_transform.creat_label_set(name='accusation')
labels = data_transform.creat_labels(name='accusation')#  案件个数*202
# print(len(big_labels),len(big_labels[0]),big_labels[0])
np.save('./data_deal/data_model_use/labels/' + original_dataname + '_labels_accusation.npy', labels)#形状  案件个数 *  202