## LegalCap project
### author: ezra 
### data 20190619
****

1. 数据预处理再data_preprocessing下面，原始数据放在data_original文件夹下面
+ 其他函数的数据预处理都在data_preprocessing文件下面的，比如：分词（data_cut.py），one-hot标签建立（data_label.py），填充词生成索引（fit_tokenizer_to_sequences.py）
+ svm 的数据预处理的入口文件是 svm_data
2. 数据增强
