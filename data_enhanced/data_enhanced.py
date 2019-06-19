import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 罪名数量分布
labels = np.load('../data_preprocessing/data_deal/labels/big_labels_accusation.npy')
x = labels.sum(axis=1)
for i in range(int(x.max()) + 1):
    # s1 = (x==i)
    # print('罪名数=%d的有%d个' % (i, s1.sum()))
    print('罪名数=%d的有%d个' % (i, (x == i).sum()))
    #一个案件共有几个罪名的统计  列出来
######################################################################################################
# 不平衡修正，记录少类别的索引
path = '../data_preprocessing/data/data_all.json'
f = open(path, 'r', encoding='utf8')
line = True
data = []
n = 0
while line:
    line = f.readline()
    try:
        data.append(json.loads(line))
    except Exception as e:
        print('num: %d' % n)
        print('error: %s' % e)
        print('data: %s' % line)
        print(n)
    n += 1
    if n % 200000 == 0:
        print('finish read %d lines' % n)

data_train, data_test = train_test_split(data, test_size=0.05, random_state=1)


class_name = []
class_index = []

for n, i in enumerate(data):
# for n, i in enumerate(data_train):
    # 只取罪名只有一个的样本  然后 获取罪名 和索引 索引就是第几个
    if len(i['meta']['accusation']) == 1:
        class_name.append(i['meta']['accusation'][0])
        class_index.append(n)

# 把罪名变成array数组
class_name_array = np.array(class_name)
# 把索引变成array数组
class_index_array = np.array(class_index)
np.save('../data_preprocessing/data_deal/class/name_accusation.npy', class_name_array)#  只取了一个罪名的样本["故意伤害","故意伤害",""......]
np.save('../data_preprocessing/data_deal/class/index_accusation.npy', class_index_array)# 刚才样本在原始数据对应的索引

maxcount = 2000
num = 10
accusation_set = list(set(class_name))#去除class_name中的多余项  比如11条数据 就只剩下 [妨碍公务 故意伤害] 因为set是创一个无序不重复的元素集合
print("accusation的长度",len(accusation_set))
index_add = []
m = 0

count = []
csv = {"label":[], "num":[]}
for i in accusation_set:
    class_count_i = class_name.count(i)#计算i在class_name中出现的次数
    class_index_i = class_index_array[class_name_array == i]#取出来出现i这个标签的索引
    print('class_count_%20s:'%i , class_count_i)
    csv["label"].append(i)
    csv["num"].append(class_count_i)
    # count.append([str(i),class_count_i])#记住所有标签的个数
    n = max(int(maxcount / class_count_i) - 1, 0)
    print('n:', n)
    m += (min(num, n) * class_count_i)
    print("m",m)
    if n > 1:
        index_add += list(class_index_i) * min(num, n)
print(csv)

df =pd.DataFrame(csv)
df.to_csv("labelcount.csv",index=False, encoding="utf_8_sig",header=True)

np.save('../data_preprocessing/data_deal/index_add_accusation_%d_%d.npy' % (maxcount, num), np.array(index_add))

'''
['盗窃']	363225
['危险驾驶']	334103
['故意伤害']	189253
['交通肇事']	161663
['走私、贩卖、运输、制造毒品']	117768
['容留他人吸毒']	54427
['诈骗']	51507
['寻衅滋事']	32019
['信用卡诈骗']	21452
['抢劫']	21107
['妨害公务']	17386
['非法持有、私藏枪支、弹药']	17148
['非法持有毒品']	16471
['开设赌场']	15883
['掩饰、隐瞒犯罪所得、犯罪所得收益']	12470
['滥伐林木']	11174
['受贿']	10728
['故意毁坏财物']	9824
['抢夺']	8194
['赌博']	7776
['非法拘禁']	7172
['职务侵占']	7140
['故意杀人']	6969
['生产、销售假药']	6630
['合同诈骗']	6424
['贪污']	6417
['敲诈勒索']	5878
['过失致人死亡']	5740
['失火']	4975
['非法占用农用地']	4968
['生产、销售有毒、有害食品']	4639
['强奸']	4432
['虚开增值税专用发票、用于骗取出口退税、抵扣税款发票']	4427
['非法吸收公众存款']	4124
['聚众斗殴']	4101
['拒不执行判决、裁定']	4047
['行贿']	4038

'''
######################################################################################################
# 扩充数据集
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# num_words = 80000
# maxlen = 400
# kernel_size = 3
# DIM = 512
# batch_size = 256
#
# print('num_words = 80000, maxlen = 400')
#
# # fact数据集
# fact = np.load('../data_preprocessing/data_deal/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen))
# fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
# del fact
#
# # 标签数据集
# labels = np.load('./data_deal/labels/big_labels_accusation.npy')
# labels_train, labels_test = train_test_split(labels, test_size=0.05, random_state=1)
# del labels
#
# # set_accusation = np.load('./data_deal/set/set_accusation.npy')
# index_add_accusation = np.load('./data_deal/index_add_accusation.npy')
# fact_train = np.concatenate([fact_train, fact_train[index_add_accusation]], axis=0)
# labels_train = np.concatenate([labels_train, labels_train[index_add_accusation]], axis=0)
