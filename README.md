## LegalCap project
### author: ezra 
### data 20190619
****

1. 数据预处理再data_preprocessing下面，原始数据放在data_original文件夹下面
+ 其他函数的数据预处理都在data_preprocessing文件下面的，比如：分词（data_cut.py），one-hot标签建立（data_label.py），填充词生成索引（fit_tokenizer_to_sequences.py）
+ svm 的数据预处理的入口文件是 svm_data.py
2. 深度学习的数据增强在data_preprocessing文件夹里面，其中(data_enhanced.py)是生成数据增强的索引，索引放在了enhanced_index文件夹里面的，addfact.py是具体的使用索引把数据扩增结果放在了enhanced_data文件夹里面的，svm的数据增强


##### svm rsult with no enhanced
+ f1_micro: 0.7997723637258521 
+ f1_macro: 0.6444262529218608
##### svm rsult with enhanced
+ f1_micro: 0.7930047988187523  
+ f1_macro: 0.6732366823844439