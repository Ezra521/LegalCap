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
# big数据集处理
data_transform_big = data_transform()

# 读取json文件
data_transform_big.read_data(path="./data_original/"+ original_dataname +".json")

# 创建数据one-hot标签
data_transform_big.extract_data(name='accusation')
big_accusations = data_transform_big.extraction['accusation']
# print(big_accusations)
#[['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['故意伤害'], ['妨害公务', '故意伤害', '盗窃']]
data_transform_big.creat_label_set(name='accusation')
big_labels = data_transform_big.creat_labels(name='accusation')#  案件个数*202
# print(len(big_labels),len(big_labels[0]),big_labels[0])
np.save('./data_deal/data_model_use/labels/' + original_dataname + '_labels_accusation.npy', big_labels)#形状  案件个数 *  202