## LegalCap project
### author: ezra 
### data 20190619
****

1. 数据预处理再data_preprocessing下面，原始数据放在data_original文件夹下面
+ 其他函数的数据预处理都在data_preprocessing文件下面的，比如：分词（data_cut.py），one-hot标签建立（data_label.py），填充词生成索引（fit_tokenizer_to_sequences.py）
+ svm 的数据预处理的入口文件是 （svm_data.py），这个脚本主要是生成分好词的fact和label的索引，因为svm和深度学习的训练数据不一样。（data.py）是（svm_data.py）引用的一个类。（add_data_svm.py）是根据要增加list的索引数据对svm要使用的数据增强操作。
2. 深度学习的数据增强在data_preprocessing文件夹里面，其中(data_enhanced.py)是生成数据增强的索引，索引放在了enhanced_index文件夹里面的，addfact.py是具体的使用索引把数据扩增结果放在了enhanced_data文件夹里面的，svm的数据增强


##### svm rsult with no enhanced
+ f1_micro: 0.7997723637258521 
+ f1_macro: 0.6444262529218608
##### svm rsult with enhanced
+ f1_micro: 0.7930047988187523  
+ f1_macro: 0.6732366823844439

{1: 120475, 2: 30831, 3: 2914, 4: 288, 5: 58, 6: 18, 7: 5, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0, 13: 1}

CNN 50 3类别
       accu  pre_micro  recall_micro  f1_micro  pre_macro  recall_macro  f1_macro
0  0.025738   0.998902      0.416266  0.587646   0.641793      0.299872  0.394457

capsule
no 
Use tf.cast instead.
WARNING:tensorflow:From C:\dlfiles\Anaconda36\lib\site-packages\tensorflow\python\ops\math_grad.py:102: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
2019-07-03 16:45:39.831078: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_90.dll locally
C:\dlfiles\Anaconda36\lib\site-packages\sklearn\metrics\classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
C:\dlfiles\Anaconda36\lib\site-packages\sklearn\metrics\classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
      accu  pre_micro  recall_micro  f1_micro  pre_macro  recall_macro  f1_macro
0  0.73585   0.767028      0.864055  0.812656   0.729243      0.707274  0.703837
